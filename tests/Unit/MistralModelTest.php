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

use PapiAI\Mistral\MistralModel;
use PapiAI\Mistral\MistralProvider;

describe('MistralModel', function () {
    it('is the source of truth the old constants alias', function () {
        expect(MistralProvider::MODEL_MISTRAL_LARGE)->toBe(MistralModel::Large->value);
    });

    it('ships unique IDs', function () {
        $ids = array_map(fn (MistralModel $m) => $m->value, MistralModel::cases());

        expect($ids)->toBe(array_unique($ids));
    });

    it('returns null for an ID it has not heard of, rather than throwing', function () {
        expect(MistralModel::tryFrom('not-a-model'))->toBeNull();
    });
});
