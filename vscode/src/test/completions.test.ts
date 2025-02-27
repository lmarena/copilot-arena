import * as assert from 'assert';
import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { ArenaInlineCompletionProvider } from '../inlineCompletionProvider';
import { CompletionCache } from '../cache';
import { CompletionPairResponse, ArenaCompletionItem } from '../types';
import { fetchCompletionPair } from '../api';

suite('ArenaInlineCompletionProvider Test Suite', () => {
    vscode.window.showInformationMessage('Start all ArenaInlineCompletionProvider tests.');

    let tempDir: string;
    let provider: ArenaInlineCompletionProvider;
    let mockCache: CompletionCache;
    let mockDocument: vscode.TextDocument;
    let mockPosition: vscode.Position;
    let mockContext: vscode.InlineCompletionContext;
    let mockToken: vscode.CancellationToken;

    setup(() => {
        tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'vscode-extension-test-'));
        mockCache = new CompletionCache(tempDir);
        provider = new ArenaInlineCompletionProvider(mockCache);

        mockDocument = {
            getText: () => 'const x = 10;',
            uri: { scheme: 'file' },
        } as any;

        mockPosition = new vscode.Position(0, 0);

        mockContext = {
            selectedCompletionInfo: undefined,
        } as any;

        mockToken = {
            isCancellationRequested: false,
            onCancellationRequested: () => {},
        } as any;

        // Mock workspace configuration
        (vscode.workspace.getConfiguration as any) = () => ({
            get: () => true,
        });
    });

    teardown(async () => {
        await mockCache.close();
        fs.rmSync(tempDir, { recursive: true, force: true });
    });

    function getFirstCompletionItem(result: vscode.InlineCompletionItem[] | vscode.InlineCompletionList | null): vscode.InlineCompletionItem | undefined {
        if (!result) {return undefined;}
        if (Array.isArray(result)) {return result[0];}
        return result.items[0];
    }

    test('API returns 2 completionItems, both with text', async () => {
        const mockResponse: CompletionPairResponse = {
            pairId: '123',
            completionItems: [
                { completion: 'first completion', completionId: '1', userId: 'user1', timestamp: 123, prompt: 'prompt1', model: 'test' },
                { completion: 'second completion', completionId: '2', userId: 'user1', timestamp: 124, prompt: 'prompt2', model: 'test' },
            ],
        };

        mockCache.getCompletions = async () => null;
        (fetchCompletionPair as any) = async () => mockResponse;

        const result = await provider.provideInlineCompletionItems(mockDocument, mockPosition, mockContext, mockToken);

        assert.ok(result, 'Expected a non-null result');
        const firstItem = getFirstCompletionItem(result);
        assert.ok(firstItem, 'Expected at least one completion item');
        assert.ok((firstItem.insertText as string).includes('first completion'), 'Expected first completion in result');
        assert.ok((firstItem.insertText as string).includes('second completion'), 'Expected second completion in result');

        // Check if items are in the correct location
        const firstCompletion = provider.getFirstInlineCompletion();
        const secondCompletion = provider.getSecondInlineCompletion();

        assert.ok(firstCompletion, 'Expected a first completion');
        assert.ok(secondCompletion, 'Expected a second completion');

        assert.strictEqual(firstCompletion.insertText, 'first completion', 'Expected correct first completion');
        assert.strictEqual(secondCompletion.insertText, 'second completion', 'Expected correct second completion');
    });

    test('API returns 2 completionItems, only one non-empty', async () => {
        const mockResponse: CompletionPairResponse = {
            pairId: '123',
            completionItems: [
                { completion: 'non-empty completion', completionId: '1', userId: 'user1', timestamp: 123, prompt: 'prompt1', model: 'test' },
                { completion: '', completionId: '2', userId: 'user1', timestamp: 124, prompt: 'prompt2', model: 'test' },
            ],
        };

        mockCache.getCompletions = async () => null;
        (fetchCompletionPair as any) = async () => mockResponse;

        const result = await provider.provideInlineCompletionItems(mockDocument, mockPosition, mockContext, mockToken);

        assert.ok(result, 'Expected a non-null result');
        const firstItem = getFirstCompletionItem(result);
        assert.ok(firstItem, 'Expected at least one completion item');
        assert.strictEqual(firstItem.insertText, 'non-empty completion', 'Expected only non-empty completion');

        // Check if items are in the correct location
        const singleCompletion = provider.getSingleInlineCompletion();
        assert.ok(singleCompletion, 'Expected a single completion');
        assert.strictEqual(singleCompletion.insertText, 'non-empty completion', 'Expected correct single completion');
        assert.strictEqual(provider.getFirstInlineCompletion(), undefined, 'Expected no first completion');
        assert.strictEqual(provider.getSecondInlineCompletion(), undefined, 'Expected no second completion');
    });

    test('Cache returns 2 completions', async () => {
        const mockCachedCompletions: CompletionPairResponse = {
            pairId: '123',
            completionItems: [
                { completion: 'cached first', completionId: '1', userId: 'user1', timestamp: 123, prompt: 'prompt1', model: 'test' },
                { completion: 'cached second', completionId: '2', userId: 'user1', timestamp: 124, prompt: 'prompt2', model: 'test' },
            ],
        };

        mockCache.getCompletions = async () => mockCachedCompletions;

        const result = await provider.provideInlineCompletionItems(mockDocument, mockPosition, mockContext, mockToken);

        assert.ok(result, 'Expected a non-null result');
        const firstItem = getFirstCompletionItem(result);
        assert.ok(firstItem, 'Expected at least one completion item');
        assert.ok((firstItem.insertText as string).includes('cached first'), 'Expected first cached completion in result');
        assert.ok((firstItem.insertText as string).includes('cached second'), 'Expected second cached completion in result');

        // Check if items are in the correct location
        const firstCompletion = provider.getFirstInlineCompletion();
        const secondCompletion = provider.getSecondInlineCompletion();

        assert.ok(firstCompletion, 'Expected a first completion');
        assert.ok(secondCompletion, 'Expected a second completion');

        assert.strictEqual(firstCompletion.insertText, 'cached first', 'Expected correct first cached completion');
        assert.strictEqual(secondCompletion.insertText, 'cached second', 'Expected correct second cached completion');
    });

    test('Cache returns 1 completion', async () => {
        const mockCachedCompletion: CompletionPairResponse = {
            pairId: '123',
            completionItems: [
                { completion: 'single cached completion', completionId: '1', userId: 'user1', timestamp: 123, prompt: 'prompt1', model: 'test' },
            ],
        };

        mockCache.getCompletions = async () => mockCachedCompletion;

        const result = await provider.provideInlineCompletionItems(mockDocument, mockPosition, mockContext, mockToken);

        assert.ok(result, 'Expected a non-null result');
        const firstItem = getFirstCompletionItem(result);
        assert.ok(firstItem, 'Expected at least one completion item');
        assert.strictEqual(firstItem.insertText, 'single cached completion', 'Expected single cached completion');

        // Check if items are in the correct location
        const singleCompletion = provider.getSingleInlineCompletion();
        assert.ok(singleCompletion, 'Expected a single completion');
        assert.strictEqual(singleCompletion.insertText, 'single cached completion', 'Expected correct single cached completion');
        assert.strictEqual(provider.getFirstInlineCompletion(), undefined, 'Expected no first completion');
        assert.strictEqual(provider.getSecondInlineCompletion(), undefined, 'Expected no second completion');
    });
});