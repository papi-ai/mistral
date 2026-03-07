# PapiAI Mistral Provider

[![Tests](https://github.com/papi-ai/mistral/workflows/CI/badge.svg)](https://github.com/papi-ai/mistral/actions?query=workflow%3ACI)

Mistral provider for [PapiAI](https://github.com/papi-ai/papi-core) - A simple but powerful PHP library for building AI agents.

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

## Available Models

```php
MistralProvider::MODEL_MISTRAL_LARGE  // 'mistral-large-latest' (default)
MistralProvider::MODEL_MISTRAL_EMBED  // 'mistral-embed' (embeddings)
```

## Features

- Tool/function calling
- Streaming support
- Embeddings support

## License

MIT
