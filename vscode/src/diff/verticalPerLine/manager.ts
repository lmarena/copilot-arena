import * as vscode from "vscode";
import { VerticalPerLineDiffHandler } from "./handler";
import { streamDiffLines } from "../diff";
import * as path from 'path';
import { uploadEditPair } from '../../api';
import { EditPairResponse, PrivacySetting } from '../../types';


export interface VerticalDiffCodeLens {
  start: number;
  numRed: number;
  numGreen: number;
}

export class VerticalPerLineDiffManager {
  public refreshCodeLens: () => void = () => { };

  private filepathToHandler: Map<string, VerticalPerLineDiffHandler> = new Map();

  filepathToCodeLens: Map<string, VerticalDiffCodeLens[]> = new Map();

  private userChangeListeners: Map<string, vscode.Disposable> = new Map();

  private customUndoCommand: vscode.Disposable | undefined;
  private undoDisposable: vscode.Disposable | undefined;

  private originalFilepath: string | undefined;
  private tempDir: string | undefined;  // New property to store the temp directory path

  private editPairResponse: EditPairResponse | undefined;
  private documentLanguage: string | undefined;

  constructor() {
    // Remove the userChangeListener initialization
  }

  createVerticalPerLineDiffHandler(
    filepath: string,
    startLine: number,
    endLine: number,
    input: string
  ) {
    const existingHandler = this.filepathToHandler.get(filepath);
    if (existingHandler) {
      existingHandler.clear(false);
    }

    const editor = vscode.window.visibleTextEditors.find(e => e.document.uri.fsPath === filepath);
    if (editor) {
      const handler = new VerticalPerLineDiffHandler(
        startLine,
        endLine,
        editor,
        this.filepathToCodeLens,
        this.clearForFilepath.bind(this),
        this.refreshCodeLens,
        input,
      );
      this.filepathToHandler.set(filepath, handler);
      return handler;
    } else {
      return undefined;
    }
  }

  getHandlerForFile(filepath: string) {
    return this.filepathToHandler.get(filepath);
  }

  private enableDocumentChangeListener(filepath: string): vscode.Disposable | undefined {
    if (this.userChangeListeners.has(filepath)) {
      // Only create one listener per file
      return;
    }

    const listener = vscode.workspace.onDidChangeTextDocument(
      (event) => {
        if (event.document.uri.fsPath === filepath) {
          const handler = this.getHandlerForFile(filepath);
          if (handler) {
            this.handleDocumentChange(event, handler);
          }
        }
      },
    );

    this.userChangeListeners.set(filepath, listener);
    return listener;
  }

  public disableDocumentChangeListener(filepath: string) {
    const listener = this.userChangeListeners.get(filepath);
    if (listener) {
      listener.dispose();
      this.userChangeListeners.delete(filepath);
    }
  }

  private handleDocumentChange(
    event: vscode.TextDocumentChangeEvent,
    handler: VerticalPerLineDiffHandler,
  ) {
    let totalChanges = 0;
    // Loop through each change in the event
    event.contentChanges.forEach((change) => {
      // Calculate the number of lines added or removed
      const linesAdded = change.text.split("\n").length - 1;
      const linesDeleted = change.range.end.line - change.range.start.line;
      const lineDelta = linesAdded - linesDeleted;

      // Update the diff handler with the new line delta
      handler.updateLineDelta(
        event.document.uri.fsPath,
        change.range.start.line,
        lineDelta,
      );

      // Increment the total changes
      totalChanges++;
    });

    // Update the user change count in the handler
    handler.incrementUserChangeCount(totalChanges);
  }

  clearForFilepath(filepath: string | undefined, accept: boolean) {
    if (!filepath) {
      const activeEditor = vscode.window.activeTextEditor;
      if (!activeEditor) {
        return;
      }
      filepath = activeEditor.document.uri.fsPath;
    }

    const handler = this.filepathToHandler.get(filepath);
    if (handler) {
      handler.clear(accept);
      this.filepathToHandler.delete(filepath);
    }

    this.disableDocumentChangeListener(filepath);

    // Check if there are any remaining handlers
    if (this.filepathToHandler.size === 0) {
      this.unregisterCustomUndo();
    }

    vscode.commands.executeCommand("setContext", "continue.diffVisible", false);
  }

  async acceptRejectVerticalDiffBlock(
    accept: boolean,
    filepath?: string,
    index?: number,
  ) {
    if (!filepath) {
      const activeEditor = vscode.window.activeTextEditor;
      if (!activeEditor) {
        return;
      }
      filepath = activeEditor.document.uri.fsPath;
    }

    if (typeof index === "undefined") {
      index = 0;
    }

    const blocks = this.filepathToCodeLens.get(filepath);
    const block = blocks?.[index];
    if (!blocks || !block) {
      return;
    }

    const handler = this.getHandlerForFile(filepath);
    if (!handler) {
      return;
    }

    // Disable listening to file changes while continue makes changes
    this.disableDocumentChangeListener(filepath);

    // CodeLens object removed from editorToVerticalDiffCodeLens here
    await handler.acceptRejectBlock(
      accept,
      block.start,
      block.numGreen,
      block.numRed,
    );

    this.clearForFilepath(filepath, true);
    // if (blocks.length === 1) {
    // } else {
    //   // Re-enable listener for user changes to file
    //   this.enableDocumentChangeListener(filepath);
    // }
  }

  /**
   * Streams an edit to the current document based on user input and model output.
   *
   * @param input - The user's input or instruction for the edit.
   * @param modelTitle - The title of the language model to be used.
   * @param [onlyOneInsertion] - Optional flag to limit the edit to a single insertion.
   * @param [quickEdit] - Optional string indicating if this is a quick edit.
   * @param [range] - Optional range to use instead of the highlighted text. Note that the `quickEdit`
   *                  property currently can't be passed with `range` since it assumes there is an
   *                  active selection.
   *
   * This method performs the following steps:
   * 1. Sets up the editor context for the diff.
   * 2. Determines the range of text to be edited.
   * 3. Clears any existing diff handlers for the file.
   * 4. Creates a new vertical diff handler.
   * 5. Prepares the context (prefix and suffix) for the language model.
   * 6. Streams the diff lines from the language model.
   * 7. Applies the changes to the document.
   * 8. Sets up a listener for subsequent user edits.
   *
   * The method handles various edge cases, such as quick edits and existing diffs,
   * and manages the lifecycle of diff handlers and document change listeners.
   */
  async streamEdit(
    input: string,
    range?: vscode.Range,
    editor?: vscode.TextEditor
  ) {
    vscode.commands.executeCommand("setContext", "continue.diffVisible", true);

    if (!editor) {
      editor = vscode.window.activeTextEditor;
    }

    if (!editor) {
      return;
    }

    const filepath = editor.document.uri.fsPath;

    let startLine, endLine: number;

    if (range) {
      startLine = range.start.line;
      endLine = range.end.line;
    } else {
      startLine = editor.selection.start.line;
      endLine = editor.selection.end.line;
    }

    // Check for existing handler in the same file
    const existingHandler = this.getHandlerForFile(filepath);

    if (existingHandler) {
      let effectiveLineDelta = existingHandler.getLineDeltaBeforeLine(startLine);
      startLine += effectiveLineDelta;
      endLine += effectiveLineDelta;

      existingHandler.clear(false);
      this.disableDocumentChangeListener(filepath);
    }

    await new Promise((resolve) => {
      setTimeout(resolve, 200);
    });

    // Create new handler with determined start/end
    const diffHandler = this.createVerticalPerLineDiffHandler(
      filepath,
      startLine,
      endLine,
      input
    );

    if (!diffHandler) {
      console.warn("Issue occurred while creating new vertical diff handler");
      return;
    }

    let selectedRange = diffHandler.range;

    // Only if the selection is empty, use exact prefix/suffix instead of by line
    if (selectedRange.isEmpty) {
      selectedRange = new vscode.Range(
        editor.selection.start.with(undefined, 0),
        editor.selection.end.with(undefined, Number.MAX_SAFE_INTEGER),
      );
    }

    // const llm = await this.configHandler.llmFromTitle(modelTitle);
    const rangeContent = editor.document.getText(selectedRange);
    const prefix = editor.document.getText(
      new vscode.Range(new vscode.Position(0, 0), selectedRange.start),
    );
    const suffix = editor.document.getText(
      new vscode.Range(
        selectedRange.end,
        new vscode.Position(editor.document.lineCount, 0),
      ),
    );
    // const prefix = pruneLinesFromTop(
    //   editor.document.getText(
    //     new vscode.Range(new vscode.Position(0, 0), selectedRange.start),
    //   ),
    //   llm.contextLength / 4,
    //   llm.model,
    // );
    // const suffix = pruneLinesFromBottom(
    //   editor.document.getText(
    //     new vscode.Range(
    //       selectedRange.end,
    //       new vscode.Position(editor.document.lineCount, 0),
    //     ),
    //   ),
    //   llm.contextLength / 4,
    //   llm.model,
    // );

    if (editor.selection) {
      // Unselect the range
      editor.selection = new vscode.Selection(
        editor.selection.active,
        editor.selection.active,
      );
    }

    vscode.commands.executeCommand(
      "setContext",
      "continue.streamingDiff",
      true,
    );

    try {
      await diffHandler.run(
        streamDiffLines(
          prefix,
          rangeContent,
          suffix,
          input
        ),
      );

      // Scroll to the first edit
      this.scrollToFirstEdit(editor);

      // enable a listener for user edits to file while diff is open
      this.enableDocumentChangeListener(filepath);
    } catch (e) {
      this.disableDocumentChangeListener(filepath);
      vscode.window.showErrorMessage(`Error streaming diff: ${e}`);
    } finally {
      vscode.commands.executeCommand(
        "setContext",
        "continue.streamingDiff",
        false,
      );
    }
  }

  // Add this new method to the VerticalPerLineDiffManager class
  private scrollToFirstEdit(editor: vscode.TextEditor | undefined) {
    if (!editor) { return; }

    const filepath = editor.document.uri.fsPath;
    const blocks = this.filepathToCodeLens.get(filepath);
    if (blocks && blocks.length > 0) {
      const firstEditLine = blocks[0].start;
      const range = editor.document.lineAt(firstEditLine).range;
      editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
      editor.selection = new vscode.Selection(range.start, range.start);
    }
  }

  private registerCustomUndo() {
    if (this.undoDisposable) {
      // Custom undo is already registered
      return;
    }

    this.customUndoCommand = vscode.commands.registerCommand('arena.customUndo', async () => {
      const activeEditor = vscode.window.activeTextEditor;
      if (!activeEditor) {
        return;
      }

      const filepath = activeEditor.document.uri.fsPath;
      const handler = this.getHandlerForFile(filepath);

      if (handler) {
        await handler.performUndo();
        return;
      }

      // If no handler found, perform default undo
      await vscode.commands.executeCommand('default:undo');
    });

    this.undoDisposable = vscode.commands.registerCommand('undo', () => {
      vscode.commands.executeCommand('arena.customUndo');
    });
  }

  private unregisterCustomUndo() {
    if (this.customUndoCommand) {
      this.customUndoCommand.dispose();
      this.customUndoCommand = undefined;
    }
    if (this.undoDisposable) {
      this.undoDisposable.dispose();
      this.undoDisposable = undefined;
    }
  }

  async streamSideBySideDiff(
    originalUri: vscode.Uri,
    range: vscode.Range,
    llmResponse1: string,
    llmResponse2: string,
    editPairResponse: EditPairResponse
  ) {
    this.editPairResponse = editPairResponse;
    const document = await vscode.workspace.openTextDocument(originalUri);
    this.documentLanguage = document.languageId;

    this.originalFilepath = originalUri.fsPath;
    const originalDocument = await vscode.workspace.openTextDocument(originalUri);
    const originalContent = originalDocument.getText();

    // Create temporary files for the two LLM responses
    this.tempDir = path.join(path.dirname(originalUri.fsPath), '.arena_temp');
    await vscode.workspace.fs.createDirectory(vscode.Uri.file(this.tempDir));

    const tempFile1 = vscode.Uri.file(path.join(this.tempDir, 'llm_response1.txt'));
    const tempFile2 = vscode.Uri.file(path.join(this.tempDir, 'llm_response2.txt'));

    await vscode.workspace.fs.writeFile(tempFile1, Buffer.from(originalContent));
    await vscode.workspace.fs.writeFile(tempFile2, Buffer.from(originalContent));

    // Register custom undo before streaming edits
    this.registerCustomUndo();

    // Open the temporary files in new editors
    const editor1 = await vscode.window.showTextDocument(tempFile1, { viewColumn: vscode.ViewColumn.One });
    await this.streamEdit(llmResponse1, range, editor1);
    const editor2 = await vscode.window.showTextDocument(tempFile2, { viewColumn: vscode.ViewColumn.Two });
    await this.streamEdit(llmResponse2, range, editor2);

    // Scroll both editors to the first edit
    this.scrollToFirstEdit(editor1);
    this.scrollToFirstEdit(editor2);

    // Clean up temporary files when editors are closed
    const disposable = vscode.workspace.onDidCloseTextDocument(async (doc) => {
      if (doc.uri.fsPath === tempFile1.fsPath || doc.uri.fsPath === tempFile2.fsPath) {
        this.disableDocumentChangeListener(doc.uri.fsPath);
        if (this.tempDir) {
          await vscode.workspace.fs.delete(vscode.Uri.file(this.tempDir), { recursive: true });
        }
        disposable.dispose();

        // Unregister custom undo when all diff editors are closed
        if (!this.filepathToHandler.has(tempFile1.fsPath) && !this.filepathToHandler.has(tempFile2.fsPath)) {
          this.unregisterCustomUndo();
        }
      }
    });
  }

  async acceptLLMResponse(responseNumber: 1 | 2): Promise<EditPairResponse | undefined> {
    if (!this.originalFilepath || !this.tempDir || !this.editPairResponse) {
      return;
    }

    const llmResponse1Path = path.join(this.tempDir, 'llm_response1.txt');
    const llmResponse2Path = path.join(this.tempDir, 'llm_response2.txt');

    let selectedFilepath: string;
    let otherFilepath: string;

    if (responseNumber === 1) {
      selectedFilepath = llmResponse1Path;
      otherFilepath = llmResponse2Path;
    } else {
      selectedFilepath = llmResponse2Path;
      otherFilepath = llmResponse1Path;
    }

    // Accept the edit for the selected temporary file
    const handler = this.getHandlerForFile(selectedFilepath);
    if (handler) {
      // BUG HERE UPON SECOND TIME
      await this.acceptRejectVerticalDiffBlock(true, selectedFilepath);
      this.clearForFilepath(otherFilepath, false);
      // await this.acceptRejectVerticalDiffBlock(false, otherFilepath);
      // Save the changes in the temporary file
      const document = await vscode.workspace.openTextDocument(selectedFilepath);
      await document.save();
    }

    // Copy content from selected response to original file
    await this.copyFileContent(selectedFilepath, this.originalFilepath);

    // Clean up temporary files and handlers
    await this.cleanup(this.tempDir, [selectedFilepath, otherFilepath]);

    // Open the original file
    await this.openFile(this.originalFilepath);

    // Send the edit outcome
    await this.sendEditOutcome(responseNumber);

    // Return information about the models
    return this.editPairResponse;
  }

  async rejectAllResponses() {
    if (!this.originalFilepath || !this.tempDir) {
      return;
    }

    const llmResponse1Path = path.join(this.tempDir, 'llm_response1.txt');
    const llmResponse2Path = path.join(this.tempDir, 'llm_response2.txt');

    await this.cleanup(this.tempDir, [llmResponse1Path, llmResponse2Path]);

    // Open the original file
    await this.openFile(this.originalFilepath);

    // Send the rejection outcome
    await this.sendEditOutcome(-1); // Use -1 to indicate rejection of both responses
  }

  private async copyFileContent(sourceFilepath: string, targetFilepath: string) {
    // Read the content from the source file
    const sourceContent = await vscode.workspace.fs.readFile(vscode.Uri.file(sourceFilepath));

    // Get the TextDocument for the target file
    const targetDocument = await vscode.workspace.openTextDocument(targetFilepath);

    // Create a WorkspaceEdit to modify the target document
    const edit = new vscode.WorkspaceEdit();

    // Replace the entire content of the target document
    const fullRange = new vscode.Range(
      targetDocument.positionAt(0),
      targetDocument.positionAt(targetDocument.getText().length)
    );
    edit.replace(targetDocument.uri, fullRange, sourceContent.toString());

    // Apply the edit without saving
    await vscode.workspace.applyEdit(edit);
  }

  private async cleanup(tempDir: string, filesToDelete: string[]) {
    // Close editors without saving
    await this.closeEditors(filesToDelete);

    // Delete temporary files if they still exist
    for (const file of filesToDelete) {
      try {
        await vscode.workspace.fs.stat(vscode.Uri.file(file));
        await vscode.workspace.fs.delete(vscode.Uri.file(file));
      } catch (error) {
        // File doesn't exist, which is fine
      }
      const handler = this.filepathToHandler.get(file);
      if (handler) {
        handler.dispose();
        this.filepathToHandler.delete(file);
      }
    }

    // Remove the temporary directory if it still exists
    try {
      await vscode.workspace.fs.stat(vscode.Uri.file(tempDir));
      await vscode.workspace.fs.delete(vscode.Uri.file(tempDir), { recursive: true });
    } catch (error) {
      // Directory doesn't exist, which is fine
    }

    vscode.commands.executeCommand("setContext", "arena.streamingDiff", false);

    // Clear the tempDir property after cleanup
    this.tempDir = undefined;
  }

  private async closeEditors(filepaths: string[]) {
    for (const filepath of filepaths) {
      const document = await vscode.workspace.openTextDocument(filepath);
      // Revert the document to discard any changes
      await vscode.window.showTextDocument(document, { preserveFocus: false });

      await vscode.commands.executeCommand('workbench.action.files.revert', document.uri);
    }

    for (const filepath of filepaths) {
      const document = await vscode.workspace.openTextDocument(filepath);
      // Now close the editor
      const edit = new vscode.WorkspaceEdit();
      edit.deleteFile(document.uri, { ignoreIfNotExists: true });
      await vscode.workspace.applyEdit(edit);
    }

    // Force close any remaining tabs without saving
    const editorsToClose = vscode.window.tabGroups.all
      .flatMap(group => group.tabs)
      .filter(tab => tab.input instanceof vscode.TabInputText && filepaths.includes(tab.input.uri.fsPath));

    if (editorsToClose.length > 0) {
      await vscode.window.tabGroups.close(editorsToClose, true);
    }
  }

  private async openFile(filepath: string) {
    const document = await vscode.workspace.openTextDocument(filepath);
    await vscode.window.showTextDocument(document, { preview: false });
  }

  private async sendEditOutcome(acceptedIndex: number) {
    const config = vscode.workspace.getConfiguration("arena");
    const privacySetting = config.get<PrivacySetting>("codePrivacySettings") || PrivacySetting.Private;
    if (!this.editPairResponse || this.documentLanguage === undefined) {
      console.error('Edit pair response or document language is not available');
      return;
    }

    try {
      await uploadEditPair(
        this.editPairResponse,
        acceptedIndex - 1, // Convert to 0-based index
        privacySetting,
        this.documentLanguage
      );
      console.log('Edit outcome successfully uploaded');
    } catch (error) {
      console.error('Failed to upload edit outcome:', error);
    }
  }
}
