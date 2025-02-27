import * as assert from 'assert';
import * as vscode from 'vscode';
import { isMidSpan } from '../utils';

suite('isMidSpan Function Test Suite', () => {
    vscode.window.showInformationMessage('Start all isMidSpan function tests.');

    test('isMidSpan returns true when prefix does not end with newline', () => {
        const prefix = 'const x = 10;';
        assert.strictEqual(isMidSpan(prefix), true, 'Expected isMidSpan to return true when prefix does not end with newline');
    });

    test('isMidSpan returns false when prefix ends with newline', () => {
        const prefix = 'const x = 10;\n';
        assert.strictEqual(isMidSpan(prefix), false, 'Expected isMidSpan to return false when prefix ends with newline');
    });

    test('isMidSpan returns false when prefix ends with newline followed by spaces', () => {
        const prefix = 'const x = 10;\n    ';
        assert.strictEqual(isMidSpan(prefix), false, 'Expected isMidSpan to return false when prefix ends with newline followed by spaces');
    });

    test('isMidSpan returns true when prefix ends with spaces but no newline', () => {
        const prefix = 'const x = 10;    ';
        assert.strictEqual(isMidSpan(prefix), true, 'Expected isMidSpan to return true when prefix ends with spaces but no newline');
    });

    test('isMidSpan returns true for empty string', () => {
        const prefix = '';
        assert.strictEqual(isMidSpan(prefix), true, 'Expected isMidSpan to return true for empty string');
    });

    test('isMidSpan returns false for string with only newline', () => {
        const prefix = '\n';
        assert.strictEqual(isMidSpan(prefix), false, 'Expected isMidSpan to return false for string with only newline');
    });

    test('isMidSpan returns false for string with only spaces and newline', () => {
        const prefix = '   \n';
        assert.strictEqual(isMidSpan(prefix), false, 'Expected isMidSpan to return false for string with only spaces and newline');
    });

    test('isMidSpan returns true for string with only spaces', () => {
        const prefix = '   ';
        assert.strictEqual(isMidSpan(prefix), true, 'Expected isMidSpan to return true for string with only spaces');
    });

    test('isMidSpan returns true for multiline string not ending with newline', () => {
        const prefix = 'line1\nline2\nline3';
        assert.strictEqual(isMidSpan(prefix), true, 'Expected isMidSpan to return true for multiline string not ending with newline');
    });

    test('isMidSpan returns false for multiline string ending with newline', () => {
        const prefix = 'line1\nline2\nline3\n';
        assert.strictEqual(isMidSpan(prefix), false, 'Expected isMidSpan to return false for multiline string ending with newline');
    });
});