import * as assert from 'assert';
import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { ArenaCompletionItem, CompletionPairResponse } from '../types';
import { CompletionCache } from '../cache';

suite('Cache Test Suite', () => {
    vscode.window.showInformationMessage('Start all cache tests.');

    let tempDir: string;
    let cache: CompletionCache;

    setup(async () => {
        tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'vscode-extension-test-'));
        cache = new CompletionCache(tempDir);
        await cache.init();
    });

    teardown(async () => {
        await cache.close();
        fs.rmSync(tempDir, { recursive: true, force: true });
    });

    async function addCompletions(prefix: string, completions: [string, string, string][]) {
        for (const [completion, model, completionId] of completions) {
            const completionItem: ArenaCompletionItem = {
                completionId: completionId,
                userId: 'test-user',
                timestamp: Date.now(),
                prompt: 'test-prompt',
                completion: completion,
                model: model,
            };
            await cache.addCompletion(prefix, completionItem);
        }
    }

    function assertCompletions(result: CompletionPairResponse | null, expectedCompletions: string[], expectedModels: string[], expectedCompletionIds?: string[]) {
        assert.ok(result !== null, 'Expected a non-null result');
        assert.strictEqual(result.completionItems.length, expectedCompletions.length, `Expected ${expectedCompletions.length} outcomes`);

        const actualCompletions = result.completionItems.map(outcome => outcome.completion);
        assert.deepStrictEqual(actualCompletions, expectedCompletions, 'Incorrect completions');

        const actualModels = result.completionItems.map(outcome => outcome.model);
        assert.deepStrictEqual(actualModels, expectedModels, 'Incorrect models');

        if (expectedCompletionIds) {
            const actualCompletionIds = result.completionItems.map(outcome => outcome.completionId);
            assert.deepStrictEqual(actualCompletionIds, expectedCompletionIds, 'Incorrect completionIds');
        }
    }

    test('CompletionCache init test', async () => {
        assert.ok(cache, 'CompletionCache should be initialized');
    });

    test('CompletionCache add test', async () => {
        const prefix = 'co';
        await addCompletions(prefix, [
            ['ntinue', 'model1', 'completion-1']
        ]);
    });

    test('CompletionCache basic test', async () => {
        const prefix = 'co';
        await addCompletions(prefix, [
            ['ntinue', 'model1', 'completion-1'],
            ['met', 'model2', 'completion-2']
        ]);

        const result = await cache.getCompletions(prefix);
        assertCompletions(result, ['ntinue', 'met'], ['model1', 'model2']);
    });

    test('CompletionCache shorter prefix test', async () => {
        const prefix = 'co';
        await addCompletions(prefix, [
            ['ntinue', 'model1', 'completion-1'],
            ['met', 'model2', 'completion-2']
        ] );

        const prefixRetriev = 'c';
        const result = await cache.getCompletions(prefixRetriev);
        assertCompletions(result, ['ontinue', 'omet'], ['model1', 'model2']);
    });

    test('CompletionCache longer prefix test', async () => {
        const prefix = 'c';
        await addCompletions(prefix, [
            ['ontinue', 'model1', 'completion-1'],
            ['one', 'model2', 'completion-2']
        ] );

        const prefixRetriev = 'con';
        const result = await cache.getCompletions(prefixRetriev);
        assertCompletions(result,  ['tinue', 'e'], ['model1', 'model2']);
    });

    test('CompletionCache 3 responses', async () => {
        const prefix = 'c';
        await addCompletions(prefix, [
            ['ontinue', 'model1', 'completion-1'],
            ['one', 'model2', 'completion-2'],
            ['onnosoir', 'model3', 'completion-3']
        ] );

        const prefixRetriev = 'con';
        const result = await cache.getCompletions(prefixRetriev);
        assertCompletions(result,  ['tinue', 'e', 'nosoir'], ['model1', 'model2', 'model3']);
    });

    test('CompletionCache 1 response', async () => {
        const prefix = 'c';
        await addCompletions(prefix, [
            ['ontinue', 'model1', 'completion-1'],
            ['urry', 'model2', 'completion-2'],
            ['camping', 'model3', 'completion-3']
        ] );

        const prefixRetriev = 'co';
        const result = await cache.getCompletions(prefixRetriev);
        assertCompletions(result,  ['ntinue'], ['model1']);
    });

    test('CompletionCache removal test', async () => {
        const prefix = 'co';
        await addCompletions(prefix, [
            ['ntinue', 'model1', 'completion-1'],
            ['met', 'model2', 'completion-2'],
            ['nnect', 'model3', 'completion-3']
        ] );
    
        // Verify initial state
        let result = await cache.getCompletions(prefix);
        assertCompletions(result,  ['ntinue', 'met', 'nnect'], ['model1', 'model2', 'model3'], ['completion-1', 'completion-2', 'completion-3']);
    
        // Remove one completion
        await cache.removeCompletionById('completion-2');
    
        // Verify state after removal
        result = await cache.getCompletions(prefix);
        assertCompletions(result, ['ntinue', 'nnect'], ['model1', 'model3'], ['completion-1', 'completion-3']);
    
        // Try to remove a non-existent completion (should not throw an error)
        await cache.removeCompletionById('non-existent-id');
    
        // Verify state remains unchanged
        result = await cache.getCompletions(prefix);
        assertCompletions(result, ['ntinue', 'nnect'], ['model1', 'model3'], ['completion-1', 'completion-3']);
    
        // Remove all remaining completions
        await cache.removeCompletionById('completion-1');
        await cache.removeCompletionById('completion-3');
    
        // Verify all completions are removed
        result = await cache.getCompletions(prefix);
        assert.strictEqual(result, null, 'Expected null result after removing all completions');
    });

    test('CompletionCache fullPrefix matches, but partial prefix does not match', async () => {
        const prefix = 'zero\nfirst\nsec';
        await addCompletions(prefix, [
            ['ond', 'model1', 'completion-1'],
        ] );

        const prefixRetriev = '';
        const result = await cache.getCompletions(prefixRetriev);
        assert.strictEqual(result, null, 'Should not return completions for partial prefix');
    });
});