import * as vscode from "vscode";
import {
  DecorationTypeRangeManager,
  belowIndexDecorationType,
  greenDecorationType,
  indexDecorationType,
  redDecorationType,
} from "./decorations";
import type { VerticalDiffCodeLens } from "./manager";
import { DiffLine } from "../diff";
import { commands, Disposable } from "vscode";

export class VerticalPerLineDiffHandler implements vscode.Disposable {
  public editor: vscode.TextEditor;
  private startLine: number;
  private endLine: number;
  private currentLineIndex: number;
  private cancelled = false;
  private undoDisposable: Disposable | undefined;
  private changeCount: number = 0;
  private userChangeCount: number = 0;

  public get range(): vscode.Range {
    const startLine = Math.min(this.startLine, this.endLine);
    const endLine = Math.max(this.startLine, this.endLine);
    return new vscode.Range(startLine, 0, endLine, Number.MAX_SAFE_INTEGER);
  }

  private newLinesAdded = 0;

  public input?: string;

  constructor(
    startLine: number,
    endLine: number,
    editor: vscode.TextEditor,
    private readonly editorToVerticalDiffCodeLens: Map<
      string,
      VerticalDiffCodeLens[]
    >,
    private readonly clearForFilepath: (
      filepath: string | undefined,
      accept: boolean,
    ) => void,
    private readonly refreshCodeLens: () => void,
    input?: string,
  ) {
    this.currentLineIndex = startLine;
    this.startLine = startLine;
    this.endLine = endLine;
    this.editor = editor;
    this.input = input;

    this.redDecorationManager = new DecorationTypeRangeManager(
      redDecorationType,
      this.editor,
    );
    this.greenDecorationManager = new DecorationTypeRangeManager(
      greenDecorationType,
      this.editor,
    );

    const disposable = vscode.window.onDidChangeActiveTextEditor((editor) => {
      // When we switch away and back to this editor, need to re-draw decorations
      if (editor?.document.uri.fsPath === this.filepath) {
        this.editor = editor;
        this.redDecorationManager.applyToNewEditor(editor);
        this.greenDecorationManager.applyToNewEditor(editor);
        this.updateIndexLineDecorations();
        this.refreshCodeLens();

        // Handle any lines received while editor was closed
        this.queueDiffLine(undefined);
      }
    });
    this.disposables.push(disposable);

    // Start a new undo stop before any changes
    this.editor.edit(edit => { }, { undoStopBefore: true, undoStopAfter: false });
  }

  private get filepath() {
    return this.editor.document.uri.fsPath;
  }

  private deletionBuffer: string[] = [];
  private redDecorationManager: DecorationTypeRangeManager;
  insertedInCurrentBlock = 0;

  private async insertDeletionBuffer() {
    // Don't remove trailing whitespace line
    const totalDeletedContent = this.deletionBuffer.join("\n");
    if (
      totalDeletedContent === "" &&
      this.currentLineIndex >= this.endLine + this.newLinesAdded &&
      this.insertedInCurrentBlock === 0
    ) {
      return;
    }

    if (this.deletionBuffer.length || this.insertedInCurrentBlock > 0) {
      const blocks = this.editorToVerticalDiffCodeLens.get(this.filepath) || [];
      blocks.push({
        start: this.currentLineIndex - this.insertedInCurrentBlock,
        numRed: this.deletionBuffer.length,
        numGreen: this.insertedInCurrentBlock,
      });
      this.editorToVerticalDiffCodeLens.set(this.filepath, blocks);
    }

    if (this.deletionBuffer.length === 0) {
      this.insertedInCurrentBlock = 0;
      return;
    }

    // Insert the block of deleted lines
    await this.insertTextAboveLine(
      this.currentLineIndex - this.insertedInCurrentBlock,
      totalDeletedContent,
    );
    this.redDecorationManager.addLines(
      this.currentLineIndex - this.insertedInCurrentBlock,
      this.deletionBuffer.length,
    );
    // Shift green decorations downward
    this.greenDecorationManager.shiftDownAfterLine(
      this.currentLineIndex - this.insertedInCurrentBlock,
      this.deletionBuffer.length,
    );

    // Update line index, clear buffer
    for (let i = 0; i < this.deletionBuffer.length; i++) {
      this.incrementCurrentLineIndex();
    }
    this.deletionBuffer = [];
    this.insertedInCurrentBlock = 0;

    this.refreshCodeLens();
  }

  private incrementCurrentLineIndex() {
    this.currentLineIndex++;
    this.updateIndexLineDecorations();
  }

  private greenDecorationManager: DecorationTypeRangeManager;

  private async insertTextAboveLine(index: number, text: string) {
    await this.editor.edit(
      (editBuilder) => {
        const lineCount = this.editor.document.lineCount;
        if (index >= lineCount) {
          // Append to end of file
          editBuilder.insert(
            new vscode.Position(
              lineCount,
              this.editor.document.lineAt(lineCount - 1).text.length,
            ),
            `\n${text}`,
          );
        } else {
          editBuilder.insert(new vscode.Position(index, 0), `${text}\n`);
        }
      },
      {
        undoStopBefore: false,
        undoStopAfter: false,
      },
    );
    this.changeCount++;
  }

  private async insertLineAboveIndex(index: number, line: string) {
    await this.insertTextAboveLine(index, line);
    this.greenDecorationManager.addLine(index);
    this.newLinesAdded++;
  }

  private async deleteLinesAt(index: number, numLines = 1) {
    const startLine = new vscode.Position(index, 0);
    await this.editor.edit(
      (editBuilder) => {
        editBuilder.delete(
          new vscode.Range(startLine, startLine.translate(numLines)),
        );
      },
      {
        undoStopBefore: false,
        undoStopAfter: false,
      },
    );
    this.changeCount++;
  }

  private updateIndexLineDecorations() {
    // Highlight the line at the currentLineIndex
    // And lightly highlight all lines between that and endLine
    if (this.currentLineIndex - this.newLinesAdded >= this.endLine) {
      this.editor.setDecorations(indexDecorationType, []);
      this.editor.setDecorations(belowIndexDecorationType, []);
    } else {
      const start = new vscode.Position(this.currentLineIndex, 0);
      this.editor.setDecorations(indexDecorationType, [
        new vscode.Range(
          start,
          new vscode.Position(start.line, Number.MAX_SAFE_INTEGER),
        ),
      ]);
      const end = new vscode.Position(this.endLine, 0);
      this.editor.setDecorations(belowIndexDecorationType, [
        new vscode.Range(start.translate(1), end.translate(this.newLinesAdded)),
      ]);
    }
  }

  private clearIndexLineDecorations() {
    this.editor.setDecorations(belowIndexDecorationType, []);
    this.editor.setDecorations(indexDecorationType, []);
  }

  public getLineDeltaBeforeLine(line: number) {
    // Returns the number of lines removed from a file when the diff currently active is closed
    let totalLineDelta = 0;
    for (const range of this.greenDecorationManager
      .getRanges()
      .sort((a, b) => a.start.line - b.start.line)) {
      if (range.start.line > line) {
        break;
      }

      totalLineDelta -= range.end.line - range.start.line + 1;
    }

    return totalLineDelta;
  }

  private clearEditorToVerticalDiffCodeLens(filepath: string) {
    const blocks = this.editorToVerticalDiffCodeLens.get(filepath);
    if (blocks) {
      blocks.length = 0; // Clear the array
    }
    this.editorToVerticalDiffCodeLens.delete(filepath);
  }

  async clear(accept: boolean) {
    vscode.commands.executeCommand(
      "setContext",
      "arena.streamingDiff",
      false,
    );
    const rangesToDelete = accept
      ? this.redDecorationManager.getRanges()
      : this.greenDecorationManager.getRanges();

    this.redDecorationManager.clear();
    this.greenDecorationManager.clear();
    this.clearIndexLineDecorations();

    // Clear the array in the editorToVerticalDiffCodeLens before deleting
    this.clearEditorToVerticalDiffCodeLens(this.filepath);

    await this.editor.edit(
      (editBuilder) => {
        for (const range of rangesToDelete) {
          editBuilder.delete(
            new vscode.Range(
              range.start,
              new vscode.Position(range.end.line + 1, 0),
            ),
          );
        }
      },
      {
        undoStopAfter: false,
        undoStopBefore: false,
      },
    );

    this.cancelled = true;
    this.refreshCodeLens();
    this.dispose();

    // Restore the default undo command
    if (this.undoDisposable) {
      this.undoDisposable.dispose();
      this.undoDisposable = undefined;
    }
  }

  disposables: vscode.Disposable[] = [];

  dispose() {
    this.disposables.forEach((disposable) => disposable.dispose());
  }

  get isCancelled() {
    return this.cancelled;
  }

  private _diffLinesQueue: DiffLine[] = [];
  private _queueLock = false;

  async queueDiffLine(diffLine: DiffLine | undefined) {
    if (diffLine) {
      this._diffLinesQueue.push(diffLine);
    }

    if (this._queueLock || this.editor !== vscode.window.activeTextEditor) {
      return;
    }

    this._queueLock = true;

    while (this._diffLinesQueue.length) {
      const line = this._diffLinesQueue.shift();
      if (!line) {
        break;
      }

      try {
        await this._handleDiffLine(line);
      } catch (e) {
        // If editor is switched between calling _handleDiffLine and the edit actually being executed
        this._diffLinesQueue.push(line);
        break;
      }
    }

    this._queueLock = false;
  }

  private async _handleDiffLine(diffLine: DiffLine) {
    switch (diffLine.type) {
      case "same":
        await this.insertDeletionBuffer();
        this.incrementCurrentLineIndex();
        break;
      case "old":
        // Add to deletion buffer and delete the line for now
        this.deletionBuffer.push(diffLine.line);
        await this.deleteLinesAt(this.currentLineIndex);
        break;
      case "new":
        await this.insertLineAboveIndex(this.currentLineIndex, diffLine.line);
        this.incrementCurrentLineIndex();
        this.insertedInCurrentBlock++;
        break;
    }
  }

  async run(diffLineGenerator: AsyncGenerator<DiffLine>) {
    try {
      // As an indicator of loading
      this.updateIndexLineDecorations();

      for await (const diffLine of diffLineGenerator) {
        if (this.isCancelled) {
          return;
        }
        await this.queueDiffLine(diffLine);
      }

      // Clear deletion buffer
      await this.insertDeletionBuffer();
      this.clearIndexLineDecorations();

      this.refreshCodeLens();

      // End the undo stop after all changes
      await this.editor.edit(edit => { }, { undoStopBefore: false, undoStopAfter: true });

      // Reject on user typing
      // const listener = vscode.workspace.onDidChangeTextDocument((e) => {
      //   if (e.document.uri.fsPath === this.filepath) {
      //     this.clear(false);
      //     listener.dispose();
      //   }
      // });
    } catch (e) {
      this.clearForFilepath(this.filepath, false);
      throw e;
    }
  }

  async acceptRejectBlock(
    accept: boolean,
    startLine: number,
    numGreen: number,
    numRed: number,
  ) {
    if (numGreen > 0) {
      // Delete the editor decoration
      this.greenDecorationManager.deleteRangeStartingAt(startLine + numRed);
      if (!accept) {
        // Delete the actual lines
        await this.deleteLinesAt(startLine + numRed, numGreen);
      }
    }

    if (numRed > 0) {
      const rangeToDelete =
        this.redDecorationManager.deleteRangeStartingAt(startLine);

      if (accept) {
        // Delete the actual lines
        await this.deleteLinesAt(startLine, numRed);
      }
    }

    // Shift everything below upward
    const offset = -(accept ? numRed : numGreen);
    this.redDecorationManager.shiftDownAfterLine(startLine, offset);
    this.greenDecorationManager.shiftDownAfterLine(startLine, offset);

    // Shift the codelens objects
    this.shiftCodeLensObjects(startLine, offset);
  }

  private shiftCodeLensObjects(startLine: number, lineDelta: number) {
    const blocks = this.editorToVerticalDiffCodeLens.get(this.filepath) || [];
    const updatedBlocks = blocks.map(block => {
      if (block.start > startLine) {
        // Shift blocks that start after the deletion point
        return { ...block, start: block.start + lineDelta };
      } else if (block.start <= startLine && startLine < block.start + block.numRed + block.numGreen) {
        // The change occurred within this block
        const deletionPoint = startLine - block.start;
        let newNumRed = block.numRed;
        let newNumGreen = block.numGreen;

        if (deletionPoint < block.numRed) {
          // Deletion occurred in the red part
          newNumRed = Math.max(0, block.numRed + lineDelta);
        } else {
          // Deletion occurred in the green part
          newNumGreen = Math.max(0, block.numGreen + lineDelta);
        }

        return {
          ...block,
          numRed: newNumRed,
          numGreen: newNumGreen
        };
      }
      return block;
    }).filter(block => block.numRed > 0 || block.numGreen > 0);

    this.editorToVerticalDiffCodeLens.set(this.filepath, updatedBlocks);
    this.refreshCodeLens();
  }

  public updateLineDelta(
    filepath: string,
    startLine: number,
    lineDelta: number,
  ) {
    // Retrieve the diff blocks for the given file
    const blocks = this.editorToVerticalDiffCodeLens.get(filepath);
    if (!blocks) {
      return;
    }

    // Update decorations
    this.redDecorationManager.updateRangesForDeletion(startLine, lineDelta);
    this.greenDecorationManager.updateRangesForDeletion(startLine, lineDelta);

    // Update code lens
    this.shiftCodeLensObjects(startLine, lineDelta);
  }

  public incrementUserChangeCount(count: number) {
    this.userChangeCount += count;
  }

  public getTotalChangeCount(): number {
    return this.changeCount + this.userChangeCount;
  }

  public async performUndo() {
    const totalChanges = this.getTotalChangeCount();

    // Perform undo for all changes made by the handler and user
    for (let i = 0; i < totalChanges; i++) {
      await vscode.commands.executeCommand('default:undo');
    }

    // Reset change counts
    this.changeCount = 0;
    this.userChangeCount = 0;

    // Clear all decoration ranges
    this.redDecorationManager.clear();
    this.greenDecorationManager.clear();

    // Remove all decorations from the editor
    this.editor.setDecorations(redDecorationType, []);
    this.editor.setDecorations(greenDecorationType, []);

    // Clear CodeLens for this file
    this.clearEditorToVerticalDiffCodeLens(this.filepath);
    this.refreshCodeLens();
  }
}