import { distance } from "fastest-levenshtein";
export type DiffLineType = "new" | "old" | "same";

export interface DiffLine {
    type: DiffLineType;
    line: string;
}
export type LineStream = AsyncGenerator<string>;
export type MatchLineResult = {
    /**
     * -1 if it's a new line, otherwise the index of the first match
     * in the old lines.
     */
    matchIndex: number;
    isPerfectMatch: boolean;
    newLine: string;
};
const END_BRACKETS = ["}", "});", "})"];

function linesMatchPerfectly(lineA: string, lineB: string): boolean {
    return lineA === lineB && lineA !== "";
}
function linesMatch(lineA: string, lineB: string, linesBetween = 0): boolean {
    // Require a perfect (without padding) match for these lines
    // Otherwise they are edit distance 1 from empty lines and other single char lines (e.g. each other)
    if (["}", "*", "});", "})"].includes(lineA.trim())) {
        return lineA.trim() === lineB.trim();
    }

    const d = distance(lineA, lineB);

    return (
        // Should be more unlikely for lines to fuzzy match if they are further away
        (d / Math.max(lineA.length, lineB.length) <=
            Math.max(0, 0.48 - linesBetween * 0.06) ||
            lineA.trim() === lineB.trim()) &&
        lineA.trim() !== ""
    );
}

/**
 * Used to find a match for a new line in an array of old lines.
 *
 * Return the index of the first match and whether it is a perfect match
 * Also return a version of the line with correct indentation if needs fixing
 */
export function matchLine(
    newLine: string,
    oldLines: string[],
    permissiveAboutIndentation = false,
): MatchLineResult {
    // Only match empty lines if it's the next one:
    if (newLine.trim() === "" && oldLines[0]?.trim() === "") {
        return {
            matchIndex: 0,
            isPerfectMatch: true,
            newLine: newLine.trim(),
        };
    }

    const isEndBracket = END_BRACKETS.includes(newLine.trim());

    for (let i = 0; i < oldLines.length; i++) {
        // Don't match end bracket lines if too far away
        if (i > 4 && isEndBracket) {
            return { matchIndex: -1, isPerfectMatch: false, newLine };
        }

        if (linesMatchPerfectly(newLine, oldLines[i])) {
            return { matchIndex: i, isPerfectMatch: true, newLine };
        }
        if (linesMatch(newLine, oldLines[i], i)) {
            // This is a way to fix indentation, but only for sufficiently long lines to avoid matching whitespace or short lines
            if (
                newLine.trimStart() === oldLines[i].trimStart() &&
                (permissiveAboutIndentation || newLine.trim().length > 8)
            ) {
                return {
                    matchIndex: i,
                    isPerfectMatch: true,
                    newLine: oldLines[i],
                };
            }
            return { matchIndex: i, isPerfectMatch: false, newLine };
        }
    }

    return { matchIndex: -1, isPerfectMatch: false, newLine };
}

/**
 * https://blog.jcoglan.com/2017/02/12/the-myers-diff-algorithm-part-1/
 * Invariants:
 * - new + same = newLines.length
 * - old + same = oldLinesCopy.length
 * ^ (above two guarantee that all lines get represented)
 * - Lines are always output in order, at least among old and new separately
 */
export async function* streamDiff(
    oldLines: string[],
    newLines: LineStream,
): AsyncGenerator<DiffLine> {
    const oldLinesCopy = [...oldLines];

    // If one indentation mistake is made, others are likely. So we are more permissive about matching
    let seenIndentationMistake = false;

    let newLineResult = await newLines.next();

    while (oldLinesCopy.length > 0 && !newLineResult.done) {
        const { matchIndex, isPerfectMatch, newLine } = matchLine(
            newLineResult.value,
            oldLinesCopy,
            seenIndentationMistake,
        );

        if (!seenIndentationMistake && newLineResult.value !== newLine) {
            seenIndentationMistake = true;
        }

        let type: DiffLineType;

        let isLineRemoval = false;
        const isNewLine = matchIndex === -1;

        if (isNewLine) {
            type = "new";
        } else {
            // Insert all deleted lines before match
            for (let i = 0; i < matchIndex; i++) {
                yield { type: "old", line: oldLinesCopy.shift()! };
            }

            type = isPerfectMatch ? "same" : "old";
        }

        switch (type) {
            case "new":
                yield { type, line: newLine };
                break;

            case "same":
                yield { type, line: oldLinesCopy.shift()! };
                break;

            case "old":
                yield { type, line: oldLinesCopy.shift()! };

                if (oldLinesCopy[0] !== newLine) {
                    yield { type: "new", line: newLine };
                } else {
                    isLineRemoval = true;
                }

                break;

            default:
                console.error(`Error streaming diff, unrecognized diff type: ${type}`);
        }

        if (!isLineRemoval) {
            newLineResult = await newLines.next();
        }
    }

    // Once at the edge, only one choice
    if (newLineResult.done && oldLinesCopy.length > 0) {
        for (const oldLine of oldLinesCopy) {
            yield { type: "old", line: oldLine };
        }
    }

    if (!newLineResult.done && oldLinesCopy.length === 0) {
        yield { type: "new", line: newLineResult.value };
        for await (const newLine of newLines) {
            yield { type: "new", line: newLine };
        }
    }
}

export const USELESS_LINES = ["", "```"];

function isUselessLine(line: string): boolean {
    const trimmed = line.trim().toLowerCase();
    const hasUselessLine = USELESS_LINES.some(
        (uselessLine) => trimmed === uselessLine,
    );

    return hasUselessLine || trimmed.startsWith("// end");
}

export async function* filterLeadingAndTrailingNewLineInsertion(
    diffLines: AsyncGenerator<DiffLine>,
): AsyncGenerator<DiffLine> {
    let isFirst = true;
    let buffer: DiffLine[] = [];

    for await (const diffLine of diffLines) {
        const isBlankLineInsertion =
            diffLine.type === "new" && isUselessLine(diffLine.line);

        if (isFirst && isBlankLineInsertion) {
            isFirst = false;
            continue;
        }

        isFirst = false;

        if (isBlankLineInsertion) {
            buffer.push(diffLine);
        } else {
            if (diffLine.type === "old") {
                buffer = [];
            } else {
                while (buffer.length > 0) {
                    yield buffer.shift()!;
                }
            }
            yield diffLine;
        }
    }
}

async function* addIndentation(
    diffLineGenerator: AsyncGenerator<DiffLine>,
    indentation: string,
): AsyncGenerator<DiffLine> {
    for await (const diffLine of diffLineGenerator) {
        yield {
            ...diffLine,
            line: indentation + diffLine.line,
        };
    }
}

export async function* streamDiffLines(
    prefix: string,
    highlighted: string,
    suffix: string,
    llmOutput: string,
    onlyOneInsertion?: boolean,
): AsyncGenerator<DiffLine> {
    // Remove Telemetry capture as we no longer have llm object

    // Strip common indentation for the LLM, then add back after generation
    let oldLines =
        highlighted.length > 0
            ? highlighted.split("\n")
            : // When highlighted is empty, we need to combine last line of prefix and first line of suffix to determine the line being edited
            [(prefix + suffix).split("\n")[prefix.split("\n").length - 1]];

    // But if that line is empty, we can assume we are insertion-only
    if (oldLines.length === 1 && oldLines[0].trim() === "") {
        oldLines = [];
    }

    // Trim end of oldLines, otherwise we have trailing \r on every line for CRLF files
    oldLines = oldLines.map((line) => line.trimEnd());

    // Remove prompt construction and LLM-specific logic

    // Use llmOutput directly instead of streaming from LLM
    // let lines = streamLines(async function* () {
    //   yield llmOutput;
    // }());

    // lines = filterEnglishLinesAtStart(lines);
    // lines = filterCodeBlockLines(lines);
    // lines = stopAtLines(lines, () => { });
    // lines = skipLines(lines);

    // Remove inept-specific logic
    const lines: LineStream = (async function* () {
        for (const line of llmOutput.split('\n')) {
            yield line;
        }
    })();

    let diffLines = streamDiff(oldLines, lines);
    diffLines = filterLeadingAndTrailingNewLineInsertion(diffLines);
    if (highlighted.length === 0) {
        const line = prefix.split("\n").slice(-1)[0];
        const indentation = line.slice(0, line.length - line.trimStart().length);
        diffLines = addIndentation(diffLines, indentation);
    }

    let seenGreen = false;
    for await (const diffLine of diffLines) {
        yield diffLine;
        if (diffLine.type === "new") {
            seenGreen = true;
        } else if (onlyOneInsertion && seenGreen && diffLine.type === "same") {
            break;
        }
    }
}
