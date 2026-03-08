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

use PapiAI\Core\Contracts\EmbeddingProviderInterface;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\EmbeddingResponse;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use PapiAI\Mistral\MistralProvider;

/**
 * Test subclass that stubs HTTP methods for unit testing.
 */
class TestableMistralProvider extends MistralProvider
{
    public array $lastPayload = [];
    public array $fakeResponse = [];
    public array $fakeStreamEvents = [];
    public array $fakeEmbeddingResponse = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return $this->fakeResponse;
    }

    protected function streamRequest(array $payload): Generator
    {
        $this->lastPayload = $payload;

        foreach ($this->fakeStreamEvents as $event) {
            yield $event;
        }
    }

    protected function embeddingRequest(array $payload): array
    {
        $this->lastPayload = $payload;

        return $this->fakeEmbeddingResponse;
    }

    public function callThrowForStatusCode(int $httpCode, ?array $data): never
    {
        $this->throwForStatusCode($httpCode, $data);
    }
}

describe('MistralProvider', function () {
    beforeEach(function () {
        $this->provider = new TestableMistralProvider('test-api-key');
    });

    describe('construction', function () {
        it('implements ProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(ProviderInterface::class);
        });

        it('implements EmbeddingProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(EmbeddingProviderInterface::class);
        });

        it('returns mistral as name', function () {
            expect($this->provider->getName())->toBe('mistral');
        });
    });

    describe('capabilities', function () {
        it('supports tools', function () {
            expect($this->provider->supportsTool())->toBeTrue();
        });

        it('supports vision', function () {
            expect($this->provider->supportsVision())->toBeTrue();
        });

        it('supports structured output', function () {
            expect($this->provider->supportsStructuredOutput())->toBeTrue();
        });
    });

    describe('chat', function () {
        it('sends messages and returns a Response', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'Hello back!'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'mistral-large-latest',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $response = $this->provider->chat([Message::user('Hello')]);

            expect($response)->toBeInstanceOf(Response::class);
            expect($response->text)->toBe('Hello back!');
        });

        it('includes system message in messages array', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'OK'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'mistral-large-latest',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([
                Message::system('Be helpful'),
                Message::user('Hello'),
            ]);

            expect($this->provider->lastPayload['messages'])->toHaveCount(2);
            expect($this->provider->lastPayload['messages'][0]['role'])->toBe('system');
            expect($this->provider->lastPayload['messages'][0]['content'])->toBe('Be helpful');
            expect($this->provider->lastPayload['messages'][1]['role'])->toBe('user');
        });

        it('uses default model', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'OK'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'mistral-large-latest',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([Message::user('Hello')]);

            expect($this->provider->lastPayload['model'])->toBe('mistral-large-latest');
        });

        it('overrides model and options from parameters', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'OK'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'mistral-small-latest',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([Message::user('Hello')], [
                'model' => 'mistral-small-latest',
                'maxTokens' => 8192,
                'temperature' => 0.5,
                'stopSequences' => ['END'],
            ]);

            expect($this->provider->lastPayload['model'])->toBe('mistral-small-latest');
            expect($this->provider->lastPayload['max_tokens'])->toBe(8192);
            expect($this->provider->lastPayload['temperature'])->toBe(0.5);
            expect($this->provider->lastPayload['stop'])->toBe(['END']);
        });

        it('includes tools in payload converted to OpenAI format', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'OK'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'mistral-large-latest',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $tools = [
                [
                    'name' => 'get_weather',
                    'description' => 'Get weather',
                    'input_schema' => ['type' => 'object', 'properties' => []],
                ],
            ];

            $this->provider->chat([Message::user('Hello')], ['tools' => $tools]);

            $expected = [
                [
                    'type' => 'function',
                    'function' => [
                        'name' => 'get_weather',
                        'description' => 'Get weather',
                        'parameters' => ['type' => 'object', 'properties' => []],
                    ],
                ],
            ];
            expect($this->provider->lastPayload['tools'])->toBe($expected);
        });

        it('converts tool result messages', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'The weather is sunny'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'mistral-large-latest',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([
                Message::user('What is the weather?'),
                Message::assistant('Let me check', [
                    new ToolCall('tc_1', 'get_weather', ['city' => 'London']),
                ]),
                Message::toolResult('tc_1', ['temp' => 20]),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            expect($messages)->toHaveCount(3);

            // Tool result message
            $toolMsg = $messages[2];
            expect($toolMsg['role'])->toBe('tool');
            expect($toolMsg['tool_call_id'])->toBe('tc_1');
        });

        it('converts assistant messages with tool calls', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'Done'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'mistral-large-latest',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([
                Message::user('Hello'),
                Message::assistant('Let me help', [
                    new ToolCall('tc_1', 'search', ['q' => 'test']),
                ]),
                Message::toolResult('tc_1', 'result'),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            $assistantMsg = $messages[1];
            expect($assistantMsg['role'])->toBe('assistant');
            expect($assistantMsg['content'])->toBe('Let me help');
            expect($assistantMsg['tool_calls'][0]['id'])->toBe('tc_1');
            expect($assistantMsg['tool_calls'][0]['type'])->toBe('function');
            expect($assistantMsg['tool_calls'][0]['function']['name'])->toBe('search');
            expect($assistantMsg['tool_calls'][0]['function']['arguments'])->toBe('{"q":"test"}');
        });

        it('handles response with tool calls', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => [
                            'role' => 'assistant',
                            'content' => 'Let me check',
                            'tool_calls' => [
                                [
                                    'id' => 'call_123',
                                    'type' => 'function',
                                    'function' => [
                                        'name' => 'get_weather',
                                        'arguments' => '{"city":"London"}',
                                    ],
                                ],
                            ],
                        ],
                        'finish_reason' => 'tool_calls',
                    ],
                ],
                'model' => 'mistral-large-latest',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 20],
            ];

            $response = $this->provider->chat([Message::user('Weather?')]);

            expect($response->hasToolCalls())->toBeTrue();
            expect($response->toolCalls)->toHaveCount(1);
            expect($response->toolCalls[0]->name)->toBe('get_weather');
            expect($response->toolCalls[0]->arguments)->toBe(['city' => 'London']);
        });

        it('converts multimodal messages with url images', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'I see a cat'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'mistral-large-latest',
                'usage' => ['prompt_tokens' => 100, 'completion_tokens' => 5],
            ];

            $this->provider->chat([
                Message::userWithImage('What is this?', 'https://example.com/cat.jpg'),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            $content = $messages[0]['content'];
            expect($content)->toBeArray();
            expect($content[0]['type'])->toBe('text');
            expect($content[0]['text'])->toBe('What is this?');
            expect($content[1]['type'])->toBe('image_url');
            expect($content[1]['image_url']['url'])->toBe('https://example.com/cat.jpg');
        });

        it('converts multimodal messages with base64 images', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'I see a cat'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'mistral-large-latest',
                'usage' => ['prompt_tokens' => 100, 'completion_tokens' => 5],
            ];

            $this->provider->chat([
                Message::userWithImage('What is this?', 'base64data', 'image/png'),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            $content = $messages[0]['content'];
            expect($content[1]['type'])->toBe('image_url');
            expect($content[1]['image_url']['url'])->toBe('data:image/png;base64,base64data');
        });

        it('includes output schema as json_object response_format', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => '{"name":"test"}'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'mistral-large-latest',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $schema = ['type' => 'object', 'properties' => ['name' => ['type' => 'string']]];
            $this->provider->chat([Message::user('Hello')], ['outputSchema' => $schema]);

            expect($this->provider->lastPayload['response_format'])->toBe([
                'type' => 'json_object',
            ]);
        });
    });

    describe('stream', function () {
        it('yields StreamChunk for text deltas', function () {
            $this->provider->fakeStreamEvents = [
                ['choices' => [['delta' => ['content' => 'Hello'], 'finish_reason' => null]]],
                ['choices' => [['delta' => ['content' => ' world'], 'finish_reason' => null]]],
                ['choices' => [['delta' => [], 'finish_reason' => 'stop']]],
            ];

            $chunks = [];
            foreach ($this->provider->stream([Message::user('Hi')]) as $chunk) {
                $chunks[] = $chunk;
            }

            expect($chunks)->toHaveCount(3);
            expect($chunks[0])->toBeInstanceOf(StreamChunk::class);
            expect($chunks[0]->text)->toBe('Hello');
            expect($chunks[1]->text)->toBe(' world');
            expect($chunks[2]->isComplete)->toBeTrue();
        });

        it('sets stream flag in payload', function () {
            $this->provider->fakeStreamEvents = [
                ['choices' => [['delta' => [], 'finish_reason' => 'stop']]],
            ];

            iterator_to_array($this->provider->stream([Message::user('Hi')]));

            expect($this->provider->lastPayload['stream'])->toBeTrue();
        });

        it('ignores non-text delta events', function () {
            $this->provider->fakeStreamEvents = [
                ['choices' => [['delta' => ['role' => 'assistant'], 'finish_reason' => null]]],
                ['choices' => [['delta' => ['content' => 'Hi'], 'finish_reason' => null]]],
                ['choices' => [['delta' => [], 'finish_reason' => 'stop']]],
            ];

            $chunks = iterator_to_array($this->provider->stream([Message::user('Hi')]));

            expect($chunks)->toHaveCount(2); // text + complete
        });
    });

    describe('embed', function () {
        it('returns an EmbeddingResponse for a single input', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1, 0.2, 0.3]],
                ],
                'model' => 'mistral-embed',
                'usage' => ['prompt_tokens' => 5, 'total_tokens' => 5],
            ];

            $response = $this->provider->embed('Hello world');

            expect($response)->toBeInstanceOf(EmbeddingResponse::class);
            expect($response->embeddings)->toBe([[0.1, 0.2, 0.3]]);
            expect($response->model)->toBe('mistral-embed');
            expect($response->first())->toBe([0.1, 0.2, 0.3]);
        });

        it('returns an EmbeddingResponse for multiple inputs', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1, 0.2, 0.3]],
                    ['embedding' => [0.4, 0.5, 0.6]],
                ],
                'model' => 'mistral-embed',
                'usage' => ['prompt_tokens' => 10, 'total_tokens' => 10],
            ];

            $response = $this->provider->embed(['Hello', 'World']);

            expect($response->embeddings)->toHaveCount(2);
            expect($response->embeddings[0])->toBe([0.1, 0.2, 0.3]);
            expect($response->embeddings[1])->toBe([0.4, 0.5, 0.6]);
        });

        it('sends correct payload with default model', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1]],
                ],
                'model' => 'mistral-embed',
                'usage' => [],
            ];

            $this->provider->embed('test');

            expect($this->provider->lastPayload['model'])->toBe('mistral-embed');
            expect($this->provider->lastPayload['input'])->toBe(['test']);
        });

        it('wraps single string input as array', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1]],
                ],
                'model' => 'mistral-embed',
                'usage' => [],
            ];

            $this->provider->embed('single text');

            expect($this->provider->lastPayload['input'])->toBe(['single text']);
        });

        it('passes array input directly', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1]],
                    ['embedding' => [0.2]],
                ],
                'model' => 'mistral-embed',
                'usage' => [],
            ];

            $this->provider->embed(['text1', 'text2']);

            expect($this->provider->lastPayload['input'])->toBe(['text1', 'text2']);
        });

        it('allows overriding the model', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1]],
                ],
                'model' => 'custom-embed',
                'usage' => [],
            ];

            $this->provider->embed('test', ['model' => 'custom-embed']);

            expect($this->provider->lastPayload['model'])->toBe('custom-embed');
        });

        it('includes usage information', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1]],
                ],
                'model' => 'mistral-embed',
                'usage' => ['prompt_tokens' => 5, 'total_tokens' => 5],
            ];

            $response = $this->provider->embed('test');

            expect($response->getPromptTokens())->toBe(5);
            expect($response->getTotalTokens())->toBe(5);
        });
    });

    describe('error mapping', function () {
        it('throws AuthenticationException on 401', function () {
            $this->provider->callThrowForStatusCode(401, ['error' => ['message' => 'Invalid key']]);
        })->throws(PapiAI\Core\Exception\AuthenticationException::class);

        it('throws RateLimitException on 429', function () {
            $this->provider->callThrowForStatusCode(429, ['error' => ['message' => 'Too many requests']]);
        })->throws(PapiAI\Core\Exception\RateLimitException::class);

        it('throws ProviderException on 500', function () {
            $this->provider->callThrowForStatusCode(500, ['error' => ['message' => 'Server error']]);
        })->throws(PapiAI\Core\Exception\ProviderException::class);

        it('throws ProviderException with unknown error for null data', function () {
            $this->provider->callThrowForStatusCode(500, null);
        })->throws(PapiAI\Core\Exception\ProviderException::class);
    });
});
