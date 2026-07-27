# Mistral

Mistral provider for PapiAI.

## Installation

```bash
composer require papi-ai/mistral
```

## Usage

```php
use PapiAI\Core\Agent;
use PapiAI\Mistral\MistralProvider;

$provider = new MistralProvider(
    apiKey: $_ENV['MISTRAL_API_KEY'],
);

$agent = new Agent(
    provider: $provider,
    instructions: 'You are a helpful assistant.',
);

$response = $agent->run('Hello!');
echo $response->text;
```

## Models

```php
MistralProvider::MODEL_MISTRAL_LARGE  // 'mistral-large-latest' (default)
MistralProvider::MODEL_MISTRAL_EMBED  // 'mistral-embed' (embeddings)
```

## Capabilities

| Capability | Supported |
|---|---|
| Chat | Yes |
| Streaming | Yes |
| Tool calling | Yes |
| Vision | Yes |
| Structured output | Yes |
| Embeddings | Yes |

## Requirements

- PHP 8.2+
- `ext-curl`
- `papi-ai/papi-core` ^0.14
