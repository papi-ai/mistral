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

/**
 * Every Mistral model this package knows. All are aliases that follow their generation forward.
 *
 * The enum is the source of truth: the `MODEL_*` constants on MistralProvider alias its values, so both
 * spell the same string. It lets a watchdog enumerate what we ship instead of parsing source, and
 * each case knows whether it has been retired and what replaces it.
 *
 * An ID we have not heard of is not an error: `tryFrom()` returns null and callers may pass it
 * straight through, since next month's model is far likelier than last year's.
 *
 * @see https://docs.mistral.ai/models
 */
enum MistralModel: string
{
    case Large = 'mistral-large-latest';
    case Medium = 'mistral-medium-latest';
    case Embed = 'mistral-embed';

    /**
     * Whether the provider has retired this model.
     */
    public function isDeprecated(): bool
    {
        return false;
    }

    /**
     * The published retirement date, ISO formatted, where the provider gave one.
     */
    public function retiredOn(): ?string
    {
        return match ($this) {
            default => null,
        };
    }

    /**
     * What to use instead, for retired models that have a successor here.
     */
    public function replacement(): ?self
    {
        return match ($this) {
            default => null,
        };
    }
}
