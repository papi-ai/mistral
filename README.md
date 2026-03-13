# PapiAI Mistral Provider

[![CI](https://github.com/papi-ai/mistral/workflows/CI/badge.svg)](https://github.com/papi-ai/mistral/actions?query=workflow%3ACI) [![Latest Version](https://img.shields.io/packagist/v/papi-ai/mistral.svg)](https://packagist.org/packages/papi-ai/mistral) [![Total Downloads](https://img.shields.io/packagist/dt/papi-ai/mistral.svg)](https://packagist.org/packages/papi-ai/mistral) [![PHP Version](https://img.shields.io/packagist/php-v/papi-ai/mistral.svg)](https://packagist.org/packages/papi-ai/mistral) [![License](https://img.shields.io/packagist/l/papi-ai/mistral.svg)](https://packagist.org/packages/papi-ai/mistral)

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
