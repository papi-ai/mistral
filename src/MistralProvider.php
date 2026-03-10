<?php

/*
 * This file is part of PapiAI,
 * A simple but powerful PHP library for building AI agents.
 *
 * (c) Marcello Duarte <marcello.duarte@gmail.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

declare(strict_types=1);

namespace PapiAI\Mistral;

use Generator;
use PapiAI\Core\Contracts\EmbeddingProviderInterface;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\EmbeddingResponse;
use PapiAI\Core\Exception\AuthenticationException;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Exception\RateLimitException;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\Role;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use RuntimeException;

/**
 * Mistral AI API provider for PapiAI.
 *
 * Bridges PapiAI's core types (Message, Response, ToolCall) with Mistral's chat completions
 * API, handling format conversion in both directions. Uses an OpenAI-compatible but distinct
 * API format. Supports chat completions, streaming, tool calling, vision (multimodal),
 * structured JSON output, and text embeddings.
 *
 * Authentication is via Bearer token in the Authorization header. All HTTP is done with
 * ext-curl directly, with no HTTP abstraction layer.
 *
 * Supported models:
 *   - mistral-large-latest (general-purpose, tool use, vision, structured output)
 *   - mistral-embed (text embeddings)
 *
 * @see https://docs.mistral.ai/api/
 */
class MistralProvider implements ProviderInterface, EmbeddingProviderInterface
{
    private const API_URL = 'https://api.mistral.ai/v1/chat/completions';
    private const EMBEDDINGS_API_URL = 'https://api.mistral.ai/v1/embeddings';

    public const MODEL_MISTRAL_LARGE = 'mistral-large-latest';
    public const MODEL_MISTRAL_EMBED = 'mistral-embed';

    /**
     * Create a new Mistral provider instance.
     *
     * @param string $apiKey          Mistral API key for Bearer token authentication
     * @param string $defaultModel    Model to use when not specified in options
     * @param int    $defaultMaxTokens Maximum output tokens when not specified in options
     */
    public function __construct(
        private readonly string $apiKey,
        private readonly string $defaultModel = self::MODEL_MISTRAL_LARGE,
        private readonly int $defaultMaxTokens = 4096,
    ) {
    }

    /**
     * Send a chat completion request to the Mistral API.
     *
     * Converts PapiAI Messages to Mistral's OpenAI-compatible format, sends the request,
     * and parses the response back into a core Response object. Supports tools, vision,
     * structured output, and custom generation parameters.
     *
     * @param array<Message> $messages Conversation history as PapiAI Message objects
     * @param array{
     *     model?: string,
     *     tools?: array,
     *     maxTokens?: int,
     *     temperature?: float,
     *     stopSequences?: array<string>,
     *     outputSchema?: array,
     * } $options Request options (model, tools, maxTokens, temperature, etc.)
     *
     * @return Response Parsed response containing text, tool calls, usage, and stop reason
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function chat(array $messages, array $options = []): Response
    {
        $payload = $this->buildPayload($messages, $options);
        $response = $this->request($payload);

        return Response::fromOpenAI($response, $messages);
    }

    /**
     * Stream a chat completion from the Mistral API using server-sent events.
     *
     * Yields StreamChunk objects as partial responses arrive. The final chunk
     * has isComplete=true. Only text content is streamed.
     *
     * @param array<Message> $messages Conversation history as PapiAI Message objects
     *
     * @return iterable<StreamChunk> Stream of text chunks, ending with a completion marker
     *
     * @throws RuntimeException When the cURL request fails
     */
    public function stream(array $messages, array $options = []): iterable
    {
        $payload = $this->buildPayload($messages, $options);
        $payload['stream'] = true;

        foreach ($this->streamRequest($payload) as $event) {
            $delta = $event['choices'][0]['delta'] ?? [];
            if (isset($delta['content'])) {
                yield new StreamChunk($delta['content']);
            }
            if (($event['choices'][0]['finish_reason'] ?? null) !== null) {
                yield new StreamChunk('', isComplete: true);
            }
        }
    }

    /**
     * Generate text embeddings via the Mistral Embeddings API.
     *
     * Converts one or more text inputs into vector representations using
     * the mistral-embed model (or a custom model specified in options).
     *
     * @param string|array<string> $input  Text string or array of strings to embed
     *
     * @return EmbeddingResponse Embedding vectors with model info and token usage
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function embed(string|array $input, array $options = []): EmbeddingResponse
    {
        $model = $options['model'] ?? self::MODEL_MISTRAL_EMBED;
        $payload = [
            'model' => $model,
            'input' => is_array($input) ? $input : [$input],
        ];

        $response = $this->embeddingRequest($payload);

        $embeddings = array_map(
            fn (array $item) => $item['embedding'],
            $response['data']
        );

        return new EmbeddingResponse(
            embeddings: $embeddings,
            model: $response['model'] ?? $model,
            usage: $response['usage'] ?? [],
        );
    }

    /**
     * Whether this provider supports tool/function calling.
     */
    public function supportsTool(): bool
    {
        return true;
    }

    /**
     * Whether this provider supports vision (multimodal image input).
     */
    public function supportsVision(): bool
    {
        return true;
    }

    /**
     * Whether this provider supports structured JSON output via response_format.
     */
    public function supportsStructuredOutput(): bool
    {
        return true;
    }

    /**
     * Get the provider identifier string.
     */
    public function getName(): string
    {
        return 'mistral';
    }

    /**
     * Build the API request payload.
     */
    private function buildPayload(array $messages, array $options): array
    {
        $apiMessages = [];

        foreach ($messages as $message) {
            if ($message instanceof Message) {
                $apiMessages[] = $this->convertMessage($message);
            }
        }

        $payload = [
            'model' => $options['model'] ?? $this->defaultModel,
            'messages' => $apiMessages,
        ];

        if (isset($options['maxTokens'])) {
            $payload['max_tokens'] = $options['maxTokens'];
        }

        if (isset($options['temperature'])) {
            $payload['temperature'] = $options['temperature'];
        }

        if (isset($options['stopSequences'])) {
            $payload['stop'] = $options['stopSequences'];
        }

        // Handle structured output / JSON mode via response_format
        if (isset($options['outputSchema'])) {
            $payload['response_format'] = [
                'type' => 'json_object',
            ];
        }

        // Handle tools
        if (isset($options['tools']) && !empty($options['tools'])) {
            $payload['tools'] = $this->convertTools($options['tools']);
        }

        return $payload;
    }

    /**
     * Convert a Message to OpenAI-compatible API format.
     */
    private function convertMessage(Message $message): array
    {
        $apiMessage = [
            'role' => $this->convertRole($message->role),
        ];

        if ($message->isTool()) {
            $apiMessage['role'] = 'tool';
            $apiMessage['content'] = $message->content;
            $apiMessage['tool_call_id'] = $message->toolCallId;
        } elseif ($message->hasToolCalls()) {
            $apiMessage['content'] = $message->getText() ?: null;
            $apiMessage['tool_calls'] = array_map(function (ToolCall $tc) {
                return [
                    'id' => $tc->id,
                    'type' => 'function',
                    'function' => [
                        'name' => $tc->name,
                        'arguments' => json_encode($tc->arguments),
                    ],
                ];
            }, $message->toolCalls);
        } elseif (is_array($message->content)) {
            $apiMessage['content'] = $this->convertMultimodalContent($message->content);
        } else {
            $apiMessage['content'] = $message->content;
        }

        return $apiMessage;
    }

    /**
     * Convert multimodal content to OpenAI-compatible format.
     */
    private function convertMultimodalContent(array $content): array
    {
        $parts = [];

        foreach ($content as $part) {
            if ($part['type'] === 'text') {
                $parts[] = ['type' => 'text', 'text' => $part['text']];
            } elseif ($part['type'] === 'image') {
                $source = $part['source'];
                if ($source['type'] === 'url') {
                    $parts[] = [
                        'type' => 'image_url',
                        'image_url' => ['url' => $source['url']],
                    ];
                } else {
                    $parts[] = [
                        'type' => 'image_url',
                        'image_url' => [
                            'url' => "data:{$source['media_type']};base64,{$source['data']}",
                        ],
                    ];
                }
            }
        }

        return $parts;
    }

    /**
     * Convert tools from PapiAI format to OpenAI-compatible format.
     */
    private function convertTools(array $tools): array
    {
        $openaiTools = [];

        foreach ($tools as $tool) {
            if (is_array($tool)) {
                $openaiTools[] = [
                    'type' => 'function',
                    'function' => [
                        'name' => $tool['name'],
                        'description' => $tool['description'],
                        'parameters' => $tool['input_schema'] ?? $tool['parameters'] ?? ['type' => 'object', 'properties' => []],
                    ],
                ];
            }
        }

        return $openaiTools;
    }

    /**
     * Convert Role to OpenAI-compatible role string.
     */
    private function convertRole(Role $role): string
    {
        return match ($role) {
            Role::System => 'system',
            Role::User => 'user',
            Role::Assistant => 'assistant',
            Role::Tool => 'tool',
        };
    }

    /**
     * Send a synchronous POST request to the Mistral chat completions endpoint.
     *
     * @param array $payload JSON-encodable request body
     *
     * @return array Decoded JSON response from the API
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
     */
    protected function request(array $payload): array
    {
        $ch = curl_init(self::API_URL);

        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey,
            ],
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("Mistral API request failed: {$error}");
        }

        $data = json_decode($response, true);

        if ($httpCode >= 400) {
            $this->throwForStatusCode($httpCode, $data);
        }

        return $data;
    }

    /**
     * Throw the appropriate exception based on HTTP status code.
     *
     * Maps HTTP 401 to AuthenticationException, 429 to RateLimitException,
     * and all other error codes to a generic ProviderException.
     *
     * @param int        $httpCode HTTP response status code
     * @param array|null $data     Decoded JSON error response body
     *
     * @throws AuthenticationException When HTTP status is 401
     * @throws RateLimitException      When HTTP status is 429
     * @throws ProviderException       For all other HTTP error codes
     */
    protected function throwForStatusCode(int $httpCode, ?array $data): never
    {
        $errorMessage = $data['error']['message'] ?? 'Unknown error';

        if ($httpCode === 401) {
            throw new AuthenticationException(
                $this->getName(),
                $httpCode,
                $data,
            );
        }

        if ($httpCode === 429) {
            throw new RateLimitException(
                $this->getName(),
                statusCode: $httpCode,
                responseBody: $data,
            );
        }

        throw new ProviderException(
            "Mistral API error ({$httpCode}): {$errorMessage}",
            $this->getName(),
            $httpCode,
            $data,
        );
    }

    /**
     * Send a streaming POST request to the Mistral chat completions endpoint.
     *
     * Buffers the full SSE response, then parses and yields individual event payloads.
     * Each yielded array is a decoded JSON event from the stream.
     *
     * @param array $payload JSON-encodable request body (must include stream=true)
     *
     * @return Generator<int, array> Decoded JSON events from the SSE stream
     */
    protected function streamRequest(array $payload): Generator
    {
        $ch = curl_init(self::API_URL);

        $buffer = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey,
            ],
            CURLOPT_WRITEFUNCTION => function ($ch, $data) use (&$buffer) {
                $buffer .= $data;

                return strlen($data);
            },
        ]);

        curl_exec($ch);
        curl_close($ch);

        // Parse SSE events
        $lines = explode("\n", $buffer);
        foreach ($lines as $line) {
            $line = trim($line);
            if (str_starts_with($line, 'data: ')) {
                $json = substr($line, 6);
                if ($json === '[DONE]') {
                    break;
                }
                $event = json_decode($json, true);
                if ($event !== null) {
                    yield $event;
                }
            }
        }
    }

    /**
     * Send a synchronous POST request to the Mistral embeddings endpoint.
     *
     * @param array $payload JSON-encodable request body with model and input fields
     *
     * @return array Decoded JSON response containing embedding vectors and usage
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
     */
    protected function embeddingRequest(array $payload): array
    {
        $ch = curl_init(self::EMBEDDINGS_API_URL);

        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey,
            ],
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("Mistral Embeddings API request failed: {$error}");
        }

        $data = json_decode($response, true);

        if ($httpCode >= 400) {
            $this->throwForStatusCode($httpCode, $data);
        }

        return $data;
    }
}
