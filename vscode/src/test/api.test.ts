import * as assert from 'assert';
import * as vscode from 'vscode';
import { ArenaCompletionItem, CompletionPairResponse, CompletionPairRequest, PrivacySetting } from '../types';
import { fetchCompletionPair, uploadArenaPair, uploadArenaCompletion } from '../api'; // Adjust the import path as needed

suite('API Test Suite', () => {
    vscode.window.showInformationMessage('Start all API tests.');

    test('fetchCompletionPair should return a valid CompletionPairResponse', async () => {
        const completionPairRequest: CompletionPairRequest = {
            pairId: 'test-pair-id',
            userId: 'test-user-id',
            prefix: 'function calculateSum(a, b) {',
            suffix: '}',
            midSpan: false,
            temperature: 0.7,
            maxTokens: 100,
            topP: 1,
            maxLines: 10,
            privacy: PrivacySetting.Research,
            modelTags: []
        };

        const abortController = new AbortController();
        const signal = abortController.signal;

        const result = await fetchCompletionPair(completionPairRequest, signal);

        assert.ok(result, 'Result should not be undefined');
        assert.strictEqual(typeof result!.pairId, 'string', 'pairId should be a string');
        assert.strictEqual(result!.completionItems.length, 2, 'Should have 2 completion items');

        result!.completionItems.forEach(item => {
            assert.strictEqual(typeof item.completionId, 'string', 'completionId should be a string');
            assert.strictEqual(typeof item.userId, 'string', 'userId should be a string');
            assert.strictEqual(typeof item.timestamp, 'number', 'timestamp should be a number');
            assert.strictEqual(typeof item.prompt, 'string', 'prompt should be a string');
            assert.strictEqual(typeof item.completion, 'string', 'completion should be a string');
            assert.strictEqual(typeof item.model, 'string', 'model should be a string');
        });
    });

    test('uploadArenaPair should successfully upload pair data', async () => {
        const completionPair: CompletionPairResponse = {
            pairId: 'test-upload-pair-id',
            completionItems: [
                {
                    completionId: 'test-completion-id-1',
                    userId: 'test-user-id',
                    timestamp: Date.now(),
                    prompt: 'Test prompt 1',
                    completion: 'Test completion 1',
                    model: 'test'
                },
                {
                    completionId: 'test-completion-id-2',
                    userId: 'test-user-id',
                    timestamp: Date.now(),
                    prompt: 'Test prompt 2',
                    completion: 'Test completion 2',
                    model: 'test'
                }
            ]
        };

        const acceptedIndex = 0;

        try {
            await uploadArenaPair(completionPair, acceptedIndex, PrivacySetting.Research, "test", '');
            assert.ok(true, 'uploadArenaPair should complete without throwing an error');
        } catch (error) {
            assert.fail(`uploadArenaPair threw an unexpected error: ${error}`);
        }
    });

    test('uploadArenaCompletion should successfully upload completion data', async () => {
        const arenaCompletionItem: ArenaCompletionItem = {
            completionId: 'test-completion-id',
            userId: 'test-user-id',
            timestamp: Date.now(),
            prompt: 'Test prompt',
            completion: 'Test completion',
            model: 'test'
        };

        try {
            await uploadArenaCompletion(arenaCompletionItem, 0, 'test-pair-completion-id', PrivacySetting.Research, "test");
            assert.ok(true, 'uploadArenaCompletion should complete without throwing an error');
        } catch (error) {
            assert.fail(`uploadArenaCompletion threw an unexpected error: ${error}`);
        }
    });
});