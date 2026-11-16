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

## Creating It by Name

An application that picks its provider from configuration can build this one without naming
its class:

```php
use PapiAI\Core\Factory\ComposerDiscovery;
use PapiAI\Core\Factory\ProviderRegistry;

$registry = new ProviderRegistry(new ComposerDiscovery());

$provider = $registry->create('mistral', [
    'api_key' => $_ENV['MISTRAL_API_KEY'],
]);
```

It takes `api_key` (required), `model` and `max_tokens`. Anything it is not given falls back to the
constructor's defaults, and a missing or unknown setting throws `ConfigurationException` before any
request is made.

## Requirements

- PHP 8.2+
- `ext-curl`
- `papi-ai/papi-core` ^0.14
