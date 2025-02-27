import initSqlJs, { Database } from 'sql.js';
import * as path from 'path';
import * as fs from 'fs';
import { randomUUID } from 'crypto';
import { CompletionPairResponse, ArenaCompletionItem } from './types';

interface CompletionRow extends ArenaCompletionItem {
    prefix: string;
    sqlTimestamp: number;
}

export class CompletionCache {
    private db: Database | null = null;
    private dbPath: string;
    private initialized: boolean = false;
    private cleanupInterval: NodeJS.Timeout | null = null;

    constructor(storagePath: string) {
        this.dbPath = path.join(storagePath, 'completions.db');
    }

    public async init(): Promise<void> {
        if (this.initialized) {
            return;
        }

        try {
            const sqlJsPath = path.dirname(require.resolve('sql.js'));
            const wasmPath = path.join(sqlJsPath, 'sql-wasm.wasm');
            const SQL = await initSqlJs({
                locateFile: () => wasmPath
            });

            let data: Uint8Array | undefined;
            if (fs.existsSync(this.dbPath)) {
                data = new Uint8Array(fs.readFileSync(this.dbPath));
            }

            this.db = new SQL.Database(data);
            await this.setup();
            await this.saveChanges();
            this.initialized = true;

            // Start the automatic cleanup
            this.startAutomaticCleanup();
        } catch (error) {
            console.error('Error initializing CompletionCache:', error);
            throw error;
        }
    }

    private async setup(): Promise<void> {
        if (!this.db) {
            throw new Error('Database not initialized');
        }

        this.db.run(`
            CREATE TABLE IF NOT EXISTS completions (
                prefix TEXT,
                completion TEXT,
                model TEXT,
                completionId TEXT,
                userId TEXT,
                prompt TEXT,
                timestamp INTEGER,
                sqlTimestamp INTEGER,
                lastAccessTime INTEGER
            );
        `);
        this.db.run(`CREATE INDEX IF NOT EXISTS idx_prefix ON completions(prefix);`);
        this.db.run(`CREATE INDEX IF NOT EXISTS idx_lastAccessTime ON completions(lastAccessTime);`);
    }

    public async addCompletion(prefix: string, completionItem: ArenaCompletionItem): Promise<void> {
        if (!this.db) {
            throw new Error('Database not initialized');
        }

        try {
            const currentTime = Date.now();
            const query = `
                INSERT INTO completions (prefix, completion, model, completionId, userId, prompt, timestamp, sqlTimestamp, lastAccessTime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            `;
            this.db.run(query, [
                prefix,
                completionItem.completion,
                completionItem.model,
                completionItem.completionId,
                completionItem.userId,
                completionItem.prompt,
                completionItem.timestamp,
                currentTime,
                currentTime
            ]);
        } catch (error) {
            console.error('Error adding completion:', error);
            throw error;
        }
        await this.saveChanges();
    }

    public async removeCompletionById(completionId: string): Promise<void> {
        if (!this.db) {
            throw new Error('Database not initialized');
        }

        const query = `
            DELETE FROM completions
            WHERE completionId = ?
        `;

        this.db.run(query, [completionId]);
        await this.saveChanges();
    }

    public async getCompletions(prefix: string): Promise<CompletionPairResponse | null> {
        if (!this.db) {
            throw new Error('Database not initialized');
        }

        try {
            const currentTime = Date.now();
            const query = `
                SELECT * FROM completions 
                WHERE ? LIKE prefix || '%' OR prefix LIKE ? || '%'
                ORDER BY 
                    CASE 
                        WHEN ? LIKE prefix || '%' THEN 1
                        ELSE 2
                    END,
                    LENGTH(prefix) DESC, 
                    sqlTimestamp ASC
            `;
            const result = this.db.exec(query, [prefix, prefix, prefix]);

            if (result.length === 0 || result[0].values.length === 0) {
                return null;
            }

            const rows = result[0].values.map((row) => ({
                prefix: row[0],
                completion: row[1],
                model: row[2],
                completionId: row[3],
                userId: row[4],
                prompt: row[5],
                timestamp: row[6],
                sqlTimestamp: row[7]
            })) as CompletionRow[];

            // Update lastAccessTime for accessed items
            const updateQuery = `
                UPDATE completions
                SET lastAccessTime = ?
                WHERE completionId IN (${rows.map(() => '?').join(',')})
            `;
            this.db.run(updateQuery, [currentTime, ...rows.map(row => row.completionId)]);

            const completionItemsWithNulls: (ArenaCompletionItem | null)[] = rows.map((row: CompletionRow) => {
                const fullCompletion = row.prefix + row.completion;
                // last line of row.prefix + first line of row.completion should start with last line of the prefix
                const numberOfLinesInRowPrefix = row.prefix.split('\n').length;
                const numberOfLinesInPrefix = prefix.split('\n').length;

                if (!fullCompletion.startsWith(prefix) || numberOfLinesInPrefix < numberOfLinesInRowPrefix) {
                    return null;
                } else {
                    const adjustedCompletion = fullCompletion.slice(prefix.length);
                    return {
                        completionId: row.completionId,
                        userId: row.userId,
                        prompt: row.prompt,
                        timestamp: row.timestamp,
                        completion: adjustedCompletion,
                        model: row.model
                    };
                }
            });

            const completionItems = completionItemsWithNulls.filter((completionItem): completionItem is ArenaCompletionItem => completionItem !== null);
            if (completionItems.length === 0) {
                return null;
            }

            const uuid = randomUUID();
            await this.saveChanges();
            return { pairId: uuid, completionItems: completionItems };
        } catch (error) {
            console.error('Error getting completions:', error);
            throw error;
        }
    }

    private async saveChanges(): Promise<void> {
        if (!this.db) {
            throw new Error('Database not initialized');
        }

        if (!fs.existsSync(this.dbPath)) {
            const dir = path.dirname(this.dbPath);
            await fs.promises.mkdir(dir, { recursive: true });
        }

        const data = this.db.export();
        await fs.promises.writeFile(this.dbPath, Buffer.from(data));
    }

    public async close(): Promise<void> {
        if (this.db) {
            await this.saveChanges();
            this.db.close();
            this.db = null;
            this.initialized = false;
        }
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
            this.cleanupInterval = null;
        }
    }

    public async clearCache(): Promise<void> {
        if (!this.db) {
            throw new Error('Database not initialized');
        }

        try {
            this.db.close();

            if (fs.existsSync(this.dbPath)) {
                await fs.promises.unlink(this.dbPath);
            }

            const SQL = await initSqlJs({
                locateFile: () => path.join(path.dirname(require.resolve('sql.js')), 'sql-wasm.wasm')
            });

            this.db = new SQL.Database();
            await this.setup();
            await this.saveChanges();

            console.log('Cache reset successfully');
        } catch (error) {
            console.error('Error resetting cache:', error);
            throw error;
        }
    }

    private startAutomaticCleanup(): void {
        // Run cleanup every minute
        this.cleanupInterval = setInterval(() => {
            this.removeUnusedItems().catch(error => {
                console.error('Error during automatic cleanup:', error);
            });
        }, 60000); // 60000 ms = 1 minute
    }

    private async removeUnusedItems(): Promise<void> {
        if (!this.db) {
            throw new Error('Database not initialized');
        }

        try {
            const oneHourAgo = Date.now() - 3600000; // 1 hour in milliseconds
            const query = `
                DELETE FROM completions
                WHERE lastAccessTime < ?
            `;
            this.db.run(query, [oneHourAgo]);
            await this.saveChanges();
            console.log('Removed unused cache items');
        } catch (error) {
            console.error('Error removing unused items:', error);
            throw error;
        }
    }
}