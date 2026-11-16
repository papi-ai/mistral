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

use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Factory\AbstractProviderFactory;
use PapiAI\Core\Factory\Setting;
use PapiAI\Core\Factory\SettingValues;

/**
 * Builds a MistralProvider from plain settings, so an application can select it by name.
 */
final class MistralProviderFactory extends AbstractProviderFactory
{
    public function name(): string
    {
        return 'mistral';
    }

    public function package(): string
    {
        return 'papi-ai/mistral';
    }

    public function settings(): array
    {
        return [
            new Setting('api_key', true, 'Mistral API key'),
            new Setting('model', false, 'Default model, used when a request names none'),
            new Setting('max_tokens', false, 'Default output ceiling, used when a request sets none'),
        ];
    }

    protected function build(SettingValues $values): ProviderInterface
    {
        return new MistralProvider(...$this->argumentsGiven([
            'apiKey' => $values->string('api_key'),
            'defaultModel' => $values->stringOrNull('model'),
            'defaultMaxTokens' => $values->intOrNull('max_tokens'),
        ]));
    }
}
