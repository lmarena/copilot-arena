import * as vscode from 'vscode';
import { CompletionPairRequest, CompletionPairResponse, ArenaCompletionItem, PrivacySetting } from './types';
import { getRangeInString } from './ranges';
import { UUID, randomUUID } from 'crypto';
import { TEMPERATURE, TOP_P, MAX_LINES, MAX_OUTPUT_TOKENS, DEBOUNCE_DELAY } from './constants';
import { CompletionCache } from './cache';
import { fetchCompletionPair, uploadArenaCompletion } from './api';
import { isMidSpan, debugLog } from './utils';

export class ArenaInlineCompletionProvider implements vscode.InlineCompletionItemProvider {
    private static debounceTimeout: NodeJS.Timeout | undefined = undefined;
    private static debouncing = false;
    private static lastUUID: string | undefined = undefined;

    public isPairSuggestionVisible: boolean = false;
    public isSingleSuggestionVisible: boolean = false;
    public isClearingCompletions: boolean = false;

    private inlineCompletionList: vscode.InlineCompletionList | undefined;
    private cache: CompletionCache;

    constructor(cache: CompletionCache) {
        this.cache = cache;
    }

    public async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
        //@ts-ignore
    ): vscode.ProviderResult<vscode.InlineCompletionItem[] | vscode.InlineCompletionList> {
        const config = vscode.workspace.getConfiguration("arena");
        const privacySetting = config.get<PrivacySetting>("codePrivacySettings") || PrivacySetting.Private;
        const fileType = document.languageId;
        // Init if not already initialized
        await this.cache.init();

        // Reset the inlineCompletionList
        // this.inlineCompletionList = undefined;

        const enableTabAutocomplete =
            vscode.workspace
                .getConfiguration("arena")
                .get<boolean>("enableTabAutocomplete") || false;
        if (token.isCancellationRequested || !enableTabAutocomplete) {
            return null;
        }

        // If the text at the range isn't a prefix of the intellisense text,
        // no completion will be displayed, regardless of what we return
        if (
            context.selectedCompletionInfo &&
            !context.selectedCompletionInfo.text.startsWith(
                document.getText(context.selectedCompletionInfo.range),
            )
        ) {
            return null;
        }

        if (this.isClearingCompletions) {
            this.isClearingCompletions = false;
            return null;
        }


        try {
            // Debounce
            const uuid = randomUUID();
            ArenaInlineCompletionProvider.lastUUID = uuid;

            const abortController = new AbortController();
            const signal = abortController.signal;
            token.onCancellationRequested(() => abortController.abort());

            // Handle notebook cells
            const pos = {
                line: position.line,
                character: position.character,
            };
            let manuallyPassFileContents: string | undefined = undefined;
            if (document.uri.scheme === "vscode-notebook-cell") {
                const notebook = vscode.workspace.notebookDocuments.find((notebook) =>
                    notebook
                        .getCells()
                        .some((cell) => cell.document.uri === document.uri),
                );
                if (notebook) {
                    const cells = notebook.getCells();
                    manuallyPassFileContents = cells
                        .map((cell) => {
                            const text = cell.document.getText();
                            if (cell.kind === vscode.NotebookCellKind.Markup) {
                                return `"""${text}"""`;
                            } else {
                                return text;
                            }
                        })
                        .join("\n\n");
                    for (const cell of cells) {
                        if (cell.document.uri === document.uri) {
                            break;
                        } else {
                            pos.line += cell.document.getText().split("\n").length + 1;
                        }
                    }
                }
            }
            // Handle commit message input box
            let manuallyPassPrefix: string | undefined = undefined;
            if (document.uri.scheme === "vscode-scm") {
                return null;
            }

            const fileContents = manuallyPassFileContents ?? document.getText();
            const fullPrefix = getRangeInString(fileContents, {
                start: { line: 0, character: 0 },
                end: context.selectedCompletionInfo?.range.start ?? pos
            }) + (context.selectedCompletionInfo?.text ?? "");

            const fileLines = fileContents.split("\n");
            const fullSuffix = getRangeInString(fileContents, {
                start: pos,
                end: { line: fileLines.length - 1, character: Number.MAX_SAFE_INTEGER },
            });

            // Check if abort signal is triggered; if it is, don't even request a model response
            if (!signal || signal.aborted) {
                return null;
            }

            const startPos = context.selectedCompletionInfo?.range.start ?? position;


            // Check if the cache has any completions
            let completionPair: CompletionPairResponse | undefined = undefined;
            const cachedCompletions = await this.cache.getCompletions(fullPrefix);
            if (cachedCompletions) {
                if (cachedCompletions.completionItems.length === 1) {
                    const arenaCompletionItem = cachedCompletions.completionItems[0];
                    const singleCompletionItem = this.getSingleCompletionItem(arenaCompletionItem, startPos);
                    this.inlineCompletionList = new vscode.InlineCompletionList([singleCompletionItem]);
                    return [singleCompletionItem];
                } else if (cachedCompletions.completionItems.length >= 2) {
                    completionPair = cachedCompletions;
                }
            }


            const cacheHit = !!completionPair;
            // If no completion is found in the cache, fetch it from the server
            if (!completionPair) {
                // Debounce
                if (ArenaInlineCompletionProvider.debouncing) {
                    ArenaInlineCompletionProvider.debounceTimeout?.refresh();
                    const lastUUID = await new Promise((resolve) =>
                        setTimeout(() => {
                            resolve(ArenaInlineCompletionProvider.lastUUID);
                        }, DEBOUNCE_DELAY),
                    );
                    if (uuid !== lastUUID) {
                        return undefined;
                    }
                } else {
                    ArenaInlineCompletionProvider.debouncing = true;
                    ArenaInlineCompletionProvider.debounceTimeout = setTimeout(async () => {
                        ArenaInlineCompletionProvider.debouncing = false;
                    }, DEBOUNCE_DELAY);

                    if (context.triggerKind !== 0) {
                        // Add a Query Delay if not manually invoked
                        // const queryDelay = config.get<number>("queryDelay") || 0;
                        const queryDelay = 0.5;
                        // const removeQueryDelay = config.get<boolean>("removeQueryDelay") || false;
                        const removeQueryDelay = false;

                        if (!removeQueryDelay) {
                            await new Promise((resolve) => setTimeout(resolve, queryDelay * 1000));
                        }
                    }
                }

                // Active invocations only
                const enableAutomaticTabAutocomplete = config.get<boolean>("enableAutomaticTabAutocomplete");
                if (!enableAutomaticTabAutocomplete && context.triggerKind !== 0) {
                    return null;
                }

                // Last chance to abort
                if (!signal || signal.aborted) {
                    return null;
                }

                const pairId: UUID = randomUUID();

                const completionPairRequest: CompletionPairRequest = {
                    pairId: pairId,
                    userId: vscode.env.machineId,
                    prefix: fullPrefix,
                    suffix: fullSuffix,
                    midSpan: isMidSpan(fullPrefix),
                    temperature: TEMPERATURE,
                    maxTokens: MAX_OUTPUT_TOKENS,
                    topP: TOP_P,
                    maxLines: config.get<number>("maxOutputLines") || MAX_LINES,
                    privacy: privacySetting,
                    modelTags: config.get<string[]>('modelTags') || []
                };
                completionPair = await fetchCompletionPair(completionPairRequest, signal);
                if (!completionPair) {
                    return null;
                }
            }

            const arenaCompletionItem1 = completionPair.completionItems[0];
            const arenaCompletionItem2 = completionPair.completionItems[1];

            debugLog('Completion Pair Response:', JSON.stringify(completionPair, null, 2));
            debugLog('Cache Hit:', cacheHit);


            const originalCompletion2 = arenaCompletionItem2.completion;
            if (context.selectedCompletionInfo) {
                arenaCompletionItem1.completion = context.selectedCompletionInfo.text + arenaCompletionItem1.completion;
                arenaCompletionItem2.completion = context.selectedCompletionInfo.text + arenaCompletionItem2.completion;
            }

            const willDisplay1 = this.willDisplay(
                context.selectedCompletionInfo,
                signal,
                arenaCompletionItem1
            );
            const willDisplay2 = this.willDisplay(
                context.selectedCompletionInfo,
                signal,
                arenaCompletionItem2
            );
            if (!willDisplay1 || !willDisplay2) {
                debugLog('Will not display completion. Display booleans:', willDisplay1, willDisplay2);
                return null;
            }

            // Handle empty completions
            const arenaCompletion1IsEmpty = !arenaCompletionItem1.completion || arenaCompletionItem1.completion.length === 0;
            const arenaCompletion2IsEmpty = !arenaCompletionItem2.completion || arenaCompletionItem2.completion.length === 0;
            if ((arenaCompletion1IsEmpty && arenaCompletion2IsEmpty)) {
                debugLog("both completions are empty");
                return null;
            } else if (arenaCompletion1IsEmpty) {
                debugLog("first completion is empty");
                const singleCompletionItem = this.getSingleCompletionItem(arenaCompletionItem2, startPos);
                this.inlineCompletionList = new vscode.InlineCompletionList([singleCompletionItem]);
                await this.cache.addCompletion(fullPrefix, arenaCompletionItem2);
                uploadArenaCompletion(arenaCompletionItem1, 0, arenaCompletionItem2.completionId, privacySetting, fileType);
                uploadArenaCompletion(arenaCompletionItem2, 1, arenaCompletionItem1.completionId, privacySetting, fileType);
                return [singleCompletionItem];
            } else if (arenaCompletion2IsEmpty) {
                debugLog("second completion is empty");
                const singleCompletionItem = this.getSingleCompletionItem(arenaCompletionItem1, startPos);
                this.inlineCompletionList = new vscode.InlineCompletionList([singleCompletionItem]);
                await this.cache.addCompletion(fullPrefix, arenaCompletionItem1);
                uploadArenaCompletion(arenaCompletionItem1, 0, arenaCompletionItem2.completionId, privacySetting, fileType);
                uploadArenaCompletion(arenaCompletionItem2, 1, arenaCompletionItem1.completionId, privacySetting, fileType);
                return [singleCompletionItem];
            }

            // Handle duplicate completions
            if (arenaCompletionItem1.completion === arenaCompletionItem2.completion) {
                const singleCompletionItem = this.getSingleCompletionItem(arenaCompletionItem1, startPos);
                this.inlineCompletionList = new vscode.InlineCompletionList([singleCompletionItem]);
                await this.cache.addCompletion(fullPrefix, arenaCompletionItem1);
                uploadArenaCompletion(arenaCompletionItem1, 0, arenaCompletionItem2.completionId, privacySetting, fileType);
                uploadArenaCompletion(arenaCompletionItem2, 1, arenaCompletionItem1.completionId, privacySetting, fileType);
                return [singleCompletionItem];
            }

            // Construct the range/text to show
            const completionRange1 = new vscode.Range(
                startPos,
                startPos.translate(0, arenaCompletionItem1.completion.length),
            );
            const completionRange2 = new vscode.Range(
                startPos,
                startPos.translate(0, arenaCompletionItem2.completion.length),
            );

            const prefixStart = fullPrefix.lastIndexOf('\n') + 1 || 0;
            const prefix = fullPrefix.slice(prefixStart) || '';

            // Get the suffix from the beginning of the suffix up to the first newline in the suffix
            let suffixEnd = fullSuffix.indexOf('\n');
            if (suffixEnd === -1) {
                suffixEnd = fullSuffix.length;
            }
            const suffix = fullSuffix.slice(0, suffixEnd);

            // Calculate the indentation after the last newline in the prefix
            const lastNewlineIndex = prefix.lastIndexOf('\n') ?? -1;
            const indentationStart = lastNewlineIndex + 1;
            const indentationEnd = prefix.length;

            let indentation = '';
            for (let i = indentationStart; i < indentationEnd; i++) {
                const char = prefix[i];
                if (char === ' ' || char === '\t') {
                    indentation += char;
                } else {
                    break;
                }
            }
            const spaceIndentation = " ".repeat(prefix.length);
            // const separator = `${spaceIndentation}======`;

            let combinedCompletionPrefix = "";
            if (!(prefix.trim().length === 0)) {
                // combinedCompletionPrefix = `\n${spaceIndentation}`;
                combinedCompletionPrefix = `\n${prefix}`;
            }

            // Option 2
            //       const combinedCompletion = 
            // `${outcome1.completion}
            // ${separator}
            // ${spaceIndentation}${originalOutcome2Completion}`;

            // Option 3
            //       const combinedCompletion = 
            // `${combinedCompletionPrefix}${outcome1.completion}
            // ${separator}
            // ${spaceIndentation}${originalOutcome2Completion}`;
            const completionId1Prefix = arenaCompletionItem1.completionId.slice(0, 6);
            const completionId2Prefix = arenaCompletionItem2.completionId.slice(0, 6);
            const debugCompletionIds = config.get<boolean>("displayCompletionIds")
                ? `${completionId1Prefix}|${completionId2Prefix}`
                : '';


            const separator = `${indentation}======`;
            // Option 4
            const combinedCompletion =
                `${suffix}${combinedCompletionPrefix}${arenaCompletionItem1.completion}
${separator} ${debugCompletionIds}
${prefix}${originalCompletion2}`;


            // Option 1
            // 
            //       const combinedCompletion = 
            // `${outcome1.completion}
            // ${separator}
            // ${prefix}${originalOutcome2Completion}`;

            const lines = combinedCompletion.split('\n');
            const combinedLength = lines.reduce((acc, line) => acc + line.length + 1, 0) - 1;

            const endPos = startPos.translate(0, combinedLength);
            const combinedCompletionRange = new vscode.Range(startPos, endPos);

            const combinedCompletionItem = new vscode.InlineCompletionItem(
                combinedCompletion,
                combinedCompletionRange
            );

            const completionItem1 = new vscode.InlineCompletionItem(
                arenaCompletionItem1.completion,
                completionRange1,
                {
                    title: "Log Autocomplete Outcome",
                    command: "arena.finishFirstOutcomeSuccess",
                    arguments: [completionPair, fileType],
                },
            );

            (completionItem1 as any).completeBracketPairs = true;

            const completionItem2 = new vscode.InlineCompletionItem(
                arenaCompletionItem2.completion,
                completionRange2,
                {
                    title: "Log Autocomplete Outcome",
                    command: "arena.finishSecondOutcomeSuccess",
                    arguments: [completionPair, fileType],
                },
            );

            (completionItem2 as any).completeBracketPairs = true;

            if (!cacheHit) {
                await this.cache.addCompletion(fullPrefix, arenaCompletionItem1);
                // Add the filetype (e.g. python, go, etc.)
                const fileType = document.languageId;
                uploadArenaCompletion(arenaCompletionItem1, 0, arenaCompletionItem2.completionId, privacySetting, fileType);
                await this.cache.addCompletion(fullPrefix, arenaCompletionItem2);
                uploadArenaCompletion(arenaCompletionItem2, 1, arenaCompletionItem1.completionId, privacySetting, fileType);
            }

            this.inlineCompletionList = new vscode.InlineCompletionList([combinedCompletionItem, completionItem1, completionItem2]);
            this.isPairSuggestionVisible = true;

            return [combinedCompletionItem];
        } catch (error) {
            console.error('Error providing completions:', error);
            return null;
        }
    }

    private getSingleCompletionItem(
        arenaCompletionItem: ArenaCompletionItem,
        startPos: vscode.Position,
    ): vscode.InlineCompletionItem {
        this.isPairSuggestionVisible = false;
        this.isSingleSuggestionVisible = true;
        const completionRange = new vscode.Range(
            startPos,
            startPos.translate(0, arenaCompletionItem.completion.length)
        );
        return new vscode.InlineCompletionItem(
            arenaCompletionItem.completion,
            completionRange,
            {
                title: "Log Autocomplete Outcome",
                command: "arena.finishSingleOutcomeSuccess",
                arguments: [arenaCompletionItem],
            }
        );
    }

    willDisplay(
        selectedCompletionInfo: vscode.SelectedCompletionInfo | undefined,
        abortSignal: AbortSignal,
        outcome: ArenaCompletionItem
    ): boolean {
        if (selectedCompletionInfo) {
            const { text, range } = selectedCompletionInfo;
            if (!outcome.completion.startsWith(text)) {
                debugLog(
                    `Won't display completion because text doesn't match: ${text}, ${outcome.completion}`,
                    range,
                );
                return false;
            }
        }

        if (abortSignal.aborted) {
            return false;
        }

        return true;
    }

    public getSingleInlineCompletion(): vscode.InlineCompletionItem | undefined {
        if (!this.inlineCompletionList || !this.isSingleSuggestionVisible) {
            return undefined;
        }
        return this.inlineCompletionList?.items[0];
    }

    public getFirstInlineCompletion(): vscode.InlineCompletionItem | undefined {
        if (!this.inlineCompletionList || !this.isPairSuggestionVisible) {
            return undefined;
        }
        return this.inlineCompletionList?.items[1];
    }

    public getSecondInlineCompletion(): vscode.InlineCompletionItem | undefined {
        if (!this.inlineCompletionList || !this.isPairSuggestionVisible) {
            return undefined;
        }
        return this.inlineCompletionList?.items[2];
    }
}