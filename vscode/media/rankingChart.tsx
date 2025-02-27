import * as React from 'react';
import * as ReactDOM from 'react-dom/client';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, TooltipProps } from 'recharts';
import { NameType, ValueType } from 'recharts/types/component/DefaultTooltipContent';
import { ModelSelection } from '../src/types';

// Color palette
const colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#82E0AA', '#F1948A', '#85C1E9'
];

interface RankingDataPoint {
    model: string;
    rank: number;
}

declare global {
    interface Window {
        acquireVsCodeApi: () => any;
    }
}

const vscode = window.acquireVsCodeApi();

const isDevelopment = false;

// Replace all console.log calls with this function
const log = (...args: any[]) => {
    if (isDevelopment) {
        console.log(...args);
    }
};

// Replace console.error calls with this function
const logError = (...args: any[]) => {
    if (isDevelopment) {
        console.error(...args);
    }
};

const RankingChart: React.FC = () => {
    log('RankingChart component initialized');

    const [data, setData] = React.useState<Array<Array<RankingDataPoint>>>([]);
    const [selectionHistory, setSelectionHistory] = React.useState<ModelSelection[]>([]);
    const [sortAscending, setSortAscending] = React.useState(false);
    const [expandedSections, setExpandedSections] = React.useState<string[]>(['pastDay', 'pastWeek', 'pastMonth', 'older']);
    const [userVoteCount, setUserVoteCount] = React.useState(0);
    const [mostRecentScores, setMostRecentScores] = React.useState<Array<{ model: string, elo: number, votes: number }>>([]);
    const [showLineChart, setShowLineChart] = React.useState(false);
    const [username, setUsername] = React.useState<string | null>(null);
    const [showRegisterModal, setShowRegisterModal] = React.useState(false);

    const CHART_HEIGHT = 400; // Fixed height for both charts
    const MIN_CHART_WIDTH = 200;
    const MIN_USER_VOTES = 20;

    React.useEffect(() => {
        log('useEffect hook triggered');

        const handleMessage = (event: MessageEvent) => {
            log('Message received:', event.data);
            const message = event.data;
            switch (message.type) {
                case 'updateChart':
                    setData(message.data);
                    vscode.postMessage({ type: 'log', message: 'Data updated in React component' });
                    break;
                case 'updateSelectionHistory':
                    setSelectionHistory(message.data);
                    vscode.postMessage({ type: 'log', message: 'Selection history updated in React component' });
                    break;
                case 'initializeSelections':
                    setSelectionHistory(message.selections);
                    setSortAscending(message.sortAscending);
                    vscode.postMessage({ type: 'log', message: 'Selections initialized in React component' });
                    break;
                case 'updateUserScores':
                    setMostRecentScores(message.data);
                    break;
                case 'updateUserVoteCount':
                    setUserVoteCount(message.data);
                    break;
                case 'loginSuccess':
                    setUsername(message.username);
                    // Trigger leaderboard update
                    vscode.postMessage({ type: 'updateLeaderboard' });
                    break;
                case 'logoutSuccess':
                    setUsername(null);
                    break;
                default:
                    logError('Unknown message type:', message.type);
            }
        };

        window.addEventListener('message', handleMessage);
        vscode.postMessage({ type: 'log', message: 'React component mounted' });
        vscode.postMessage({ type: 'webviewReady' });

        return () => {
            log('Component unmounting, removing event listener');
            window.removeEventListener('message', handleMessage);
        };
    }, []);

    const toggleSort = () => {
        setSortAscending(!sortAscending);
        vscode.postMessage({ type: 'toggleSort', sortAscending: !sortAscending });
    };

    const sortedSelectionHistory = React.useMemo(() => {
        return [...selectionHistory].sort((a, b) =>
            sortAscending ? a.timestamp - b.timestamp : b.timestamp - a.timestamp
        );
    }, [selectionHistory, sortAscending]);

    const handleShare = () => {
        const tweetText = encodeURIComponent("Check out my AI model rankings!");
        const tweetUrl = encodeURIComponent("https://marketplace.visualstudio.com/items?itemName=CodeArena.code-arena");
        const twitterUrl = `https://twitter.com/intent/tweet?text=${tweetText}&url=${tweetUrl}`;

        vscode.postMessage({ type: 'openExternalUrl', url: twitterUrl });
    };

    const handleRegister = (username: string, password?: string, metadata?: any) => {
        vscode.postMessage({ type: 'register', username, password, metadata });
        setShowRegisterModal(false);
    };

    log('Preparing chart data');
    const allModels = Array.from(new Set(data.flatMap(dataPoint => dataPoint.map(item => item.model))));
    log('All models:', allModels);

    const formattedData = data.map((dataPoint, index) => {
        const formattedPoint: { [key: string]: number | undefined, index: number } = { index };
        allModels.forEach(model => {
            const modelData = dataPoint.find(item => item.model === model);
            formattedPoint[model] = modelData && modelData.rank <= 5 ? modelData.rank : undefined;
        });
        return formattedPoint;
    });
    log('Formatted data:', formattedData);

    log('Calculating entry/exit points');
    const entryExitPoints = allModels.reduce((acc, model) => {
        acc[model] = { entries: [], exits: [] };
        formattedData.forEach((point, index) => {
            if (index > 0) {
                const prevRank = formattedData[index - 1][model];
                const currentRank = point[model];
                if (prevRank === undefined && currentRank !== undefined) {
                    acc[model].entries.push(index);
                } else if (prevRank !== undefined && currentRank === undefined) {
                    acc[model].exits.push(index - 1);
                }
            }
        });
        return acc;
    }, {} as { [key: string]: { entries: number[], exits: number[] } });

    log('EntryExitPoints calculated:', entryExitPoints);

    log('Defining CustomTooltip');
    const CustomTooltip = ({ active, payload, label }: TooltipProps<ValueType, NameType>) => {
        log('CustomTooltip rendered', { active, payload, label });
        if (active && payload && payload.length) {
            const sortedPayload = [...payload].sort((a, b) => {
                const valueA = a.value as number;
                const valueB = b.value as number;
                return valueA - valueB;
            });

            return (
                <div style={{ backgroundColor: 'white', padding: '10px', border: '1px solid #ccc' }}>
                    <p>{`Index: ${label}`}</p>
                    {sortedPayload.map((entry, index) => (
                        <p key={index} style={{ color: entry.color }}>
                            {`${entry.name}: ${entry.value}`}
                        </p>
                    ))}
                </div>
            );
        }
        return null;
    };

    log('Defining utility functions');
    const groupSelectionsByDate = (selections: ModelSelection[]) => {
        log('Grouping selections by date');
        const now = new Date().getTime();
        const day = 24 * 60 * 60 * 1000;
        const week = 7 * day;
        const month = 30 * day;

        return selections.reduce((acc, selection) => {
            const timeDiff = now - selection.timestamp;
            if (timeDiff <= day) {
                acc.pastDay.push(selection);
            } else if (timeDiff <= week) {
                acc.pastWeek.push(selection);
            } else if (timeDiff <= month) {
                acc.pastMonth.push(selection);
            } else {
                acc.older.push(selection);
            }
            return acc;
        }, { pastDay: [], pastWeek: [], pastMonth: [], older: [] } as Record<string, ModelSelection[]>);
    };

    const toggleSection = (section: string) => {
        log('Toggling section:', section);
        setExpandedSections(prev =>
            prev.includes(section)
                ? prev.filter(s => s !== section)
                : [...prev, section]
        );
    };

    const renderSelectionGroup = (title: string, selections: ModelSelection[], sectionKey: string) => {
        log('Rendering selection group:', title);
        const isExpanded = expandedSections.includes(sectionKey);
        return (
            <div key={sectionKey}>
                <h4 onClick={() => toggleSection(sectionKey)} style={{ cursor: 'pointer' }}>
                    {title} ({selections.length}) {isExpanded ? '▼' : '▶'}
                </h4>
                {isExpanded && (
                    <ul style={{ listStyleType: 'none', padding: 0 }}>
                        {selections.map((selection, index) => (
                            <li key={index} style={{ marginBottom: '10px', padding: '10px', border: '1px solid var(--vscode-panel-border)', borderRadius: '4px', backgroundColor: 'var(--vscode-editor-background)' }}>
                                <span style={{
                                    backgroundColor: selection.selectedModel === 0 ? 'var(--vscode-editor-selectionBackground)' : 'transparent',
                                    color: selection.selectedModel === 0 ? 'var(--vscode-editor-selectionForeground)' : 'var(--vscode-foreground)',
                                    padding: '2px 5px',
                                    borderRadius: '3px',
                                    marginRight: '5px'
                                }}>
                                    {selection.model0}
                                </span>
                                vs
                                <span style={{
                                    backgroundColor: selection.selectedModel === 1 ? 'var(--vscode-editor-selectionBackground)' : 'transparent',
                                    color: selection.selectedModel === 1 ? 'var(--vscode-editor-selectionForeground)' : 'var(--vscode-foreground)',
                                    padding: '2px 5px',
                                    borderRadius: '3px',
                                    marginLeft: '5px'
                                }}>
                                    {selection.model1}
                                </span>
                                <div style={{ 
                                    fontSize: '0.8em', 
                                    color: 'var(--vscode-descriptionForeground)', 
                                    marginTop: '5px',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '10px'
                                }}>
                                    {new Date(selection.timestamp).toLocaleString()}
                                    {selection.task && (
                                        <span style={{
                                            backgroundColor: 'var(--vscode-badge-background)',
                                            color: 'var(--vscode-badge-foreground)',
                                            padding: '2px 6px',
                                            borderRadius: '3px',
                                        }}>
                                            {selection.task}
                                        </span>
                                    )}
                                </div>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        );
    };

    const groupedSelections = groupSelectionsByDate(sortedSelectionHistory);
    log('Grouped selections:', groupedSelections);

    log("User Vote Count:", userVoteCount);
    const hasEnoughData = userVoteCount >= MIN_USER_VOTES;

    const toggleChart = () => {
        setShowLineChart(!showLineChart);
    };

    const renderTable = (data: Array<{ model: string, elo: number, votes: number }>) => {
        const sortedData = [...data].sort((a, b) => b.elo - a.elo);
        return (
            <table style={tableStyle}>
                <thead>
                    <tr>
                        <th style={tableHeaderStyle}>Rank</th>
                        <th style={tableHeaderStyle}>Model</th>
                        <th style={tableHeaderStyle}>ELO Score</th>
                        <th style={tableHeaderStyle}>Votes</th>
                    </tr>
                </thead>
                <tbody>
                    {sortedData.map((item, index) => (
                        <tr key={item.model} style={index % 2 === 0 ? tableRowEvenStyle : tableRowOddStyle}>
                            <td style={tableCellStyle}>{index + 1}</td>
                            <td style={{ ...tableCellStyle, color: colors[index % colors.length] }}>
                                {item.model}
                            </td>
                            <td style={tableCellStyle}>{Math.round(item.elo)}</td>
                            <td style={tableCellStyle}>{item.votes}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        );
    };

    const tableStyle: React.CSSProperties = {
        width: '100%',
        borderCollapse: 'collapse',
        fontFamily: 'var(--vscode-font-family)',
        fontSize: '14px',
    };

    const tableHeaderStyle: React.CSSProperties = {
        backgroundColor: 'var(--vscode-editor-background)',
        color: 'var(--vscode-foreground)',
        padding: '8px',
        textAlign: 'left',
        borderBottom: '2px solid var(--vscode-panel-border)'
    };

    const tableRowEvenStyle: React.CSSProperties = {
        backgroundColor: 'var(--vscode-editor-background)'
    };

    const tableRowOddStyle: React.CSSProperties = {
        backgroundColor: 'var(--vscode-editorGutter-background)'
    };

    const tableCellStyle: React.CSSProperties = {
        padding: '8px',
        borderBottom: '1px solid var(--vscode-panel-border)'
    };

    const buttonStyle: React.CSSProperties = {
        backgroundColor: 'var(--vscode-button-background)',
        color: 'var(--vscode-button-foreground)',
        border: 'none',
        padding: '6px 12px',
        cursor: 'pointer',
        fontSize: '0.9em',
        borderRadius: '2px',
        marginLeft: '10px',
    };

    const RegisterModal: React.FC<{ onRegister: (username: string, password?: string, metadata?: any) => void, onClose: () => void }> = ({ onRegister, onClose }) => {
        const [username, setUsername] = React.useState('');
        const [jobTitle, setJobTitle] = React.useState('');

        const [yearsOfExperience, setYearsOfExperience] = React.useState('');
        const [codingHoursPerWeek, setCodingHoursPerWeek] = React.useState('');
        const [industrySector, setIndustrySector] = React.useState('');

        const handleSubmit = () => {
            const metadata = {
                jobTitle,
                yearsOfExperience: yearsOfExperience ? parseInt(yearsOfExperience) : undefined,
                codingHoursPerWeek: codingHoursPerWeek ? parseInt(codingHoursPerWeek) : undefined,
                industrySector
            };
            onRegister(username, undefined, metadata);
        };

        return (
            <div style={modalOverlayStyle}>
                <div style={modalStyle}>
                    <h2 style={{ marginTop: 0, marginBottom: '20px' }}>Register</h2>
                    <input type="text" placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} style={inputStyle} />
                    <input type="text" placeholder="Job Title (optional)" value={jobTitle} onChange={(e) => setJobTitle(e.target.value)} style={inputStyle} />
                    <select value={industrySector} onChange={(e) => setIndustrySector(e.target.value)} style={inputStyle}>
                        <option value="">Select Industry Sector (optional)</option>
                        <option value="finance">Finance</option>
                        <option value="tech">Tech</option>
                        <option value="healthcare">Healthcare</option>
                        <option value="education">Education</option>
                        <option value="other">Other</option>
                    </select>
                    <input type="number" placeholder="Years of Experience (optional)" value={yearsOfExperience} onChange={(e) => setYearsOfExperience(e.target.value)} style={inputStyle} />
                    <input type="number" placeholder="Coding Hours per Week (optional)" value={codingHoursPerWeek} onChange={(e) => setCodingHoursPerWeek(e.target.value)} style={inputStyle} />
                    <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '20px' }}>
                        <button onClick={onClose} style={{ ...buttonStyle, marginRight: '10px', backgroundColor: 'var(--vscode-button-secondaryBackground)' }}>Cancel</button>
                        <button onClick={handleSubmit} style={buttonStyle}>Register</button>
                    </div>
                </div>
            </div>
        );
    };

    const modalOverlayStyle: React.CSSProperties = {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        backdropFilter: 'blur(5px)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 1000,
    }

    const modalStyle: React.CSSProperties = {
        backgroundColor: 'var(--vscode-editor-background)',
        padding: '30px',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        maxWidth: '400px',
        width: '100%',
    };


    const inputStyle: React.CSSProperties = {
        display: 'block',
        width: '100%',
        marginBottom: '15px',
        padding: '8px',
        fontSize: '14px',
        backgroundColor: 'var(--vscode-input-background)',
        color: 'var(--vscode-input-foreground)',
        border: '1px solid var(--vscode-input-border)',
        borderRadius: '4px',
        boxSizing: 'border-box',
    };

    log('Preparing to render chart');
    return (
        <div style={{ width: '100%', height: 'auto', minWidth: `${MIN_CHART_WIDTH}px` }}>
            <div style={{ marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3>{showLineChart ? 'Ranking Over Time' : 'Current Rankings'}</h3>
                <div>
                    {username ? (
                        <span>Welcome, {username}!</span>
                    ) : (
                        <button onClick={() => setShowRegisterModal(true)} style={buttonStyle}>Register</button>
                    )}
                </div>
            </div>
            {userVoteCount === 0 ? (
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    height: '200px',
                    backgroundColor: 'var(--vscode-editor-background)',
                    color: 'var(--vscode-foreground)',
                    border: '1px solid var(--vscode-panel-border)',
                    borderRadius: '4px',
                    padding: '20px',
                    textAlign: 'center'
                }}>
                    <div>
                        <h3>No leaderboard data available yet</h3>
                        <p>Start voting to see your personal leaderboard!</p>
                        <p>Register a username to see yourself on the user votes leaderboard.</p>
                    </div>
                </div>
            ) : (
                <>
                    {showLineChart ? (
                        <div style={{ height: CHART_HEIGHT, position: 'relative', overflowX: 'auto', overflowY: 'hidden' }}>
                            <div style={{ width: '100%', minWidth: `${MIN_CHART_WIDTH}px`, height: CHART_HEIGHT }}>
                                <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
                                    <LineChart
                                        data={formattedData}
                                        margin={{ top: 10, right: 30, left: 0, bottom: 40 }}
                                    >
                                        <XAxis
                                            dataKey="index"
                                            tick={{ fill: 'rgba(255, 255, 255, 0.8)' }}
                                            tickFormatter={(value) => `${(value + 1) * 20}`}
                                        />
                                        <YAxis
                                            domain={[1, 5]}
                                            reversed
                                            width={20}
                                            tick={{ fill: 'rgba(255, 255, 255, 0.8)' }}
                                        />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Legend />
                                        {allModels.map((model, index) => (
                                            <Line
                                                key={model}
                                                type="linear"
                                                dataKey={model}
                                                name={model}
                                                stroke={colors[index % colors.length]}
                                                strokeWidth={2}
                                                dot={(props: any) => {
                                                    const { cx, cy, payload, index: dotIndex } = props;
                                                    const dataIndex = payload.index;
                                                    const modelPoints = entryExitPoints[model];
                                                    const isLastPoint = dotIndex === formattedData.length - 1;
                                                    const isOnlyPoint = formattedData.length === 1;
                                                    const hasValue = payload[model] !== undefined;

                                                    if (!hasValue) {
                                                        return <circle cx={cx} cy={cy} r={0} fill="transparent" />
                                                    }

                                                    if (isLastPoint || isOnlyPoint) {
                                                        return (
                                                            <g>
                                                                <circle cx={cx} cy={cy} r={6} fill={colors[index % colors.length]} />
                                                                <circle cx={cx} cy={cy} r={3} fill="white" />
                                                            </g>
                                                        );
                                                    } else if (modelPoints.entries.includes(dataIndex)) {
                                                        return <circle cx={cx} cy={cy} r={4} fill={colors[index % colors.length]} />;
                                                    } else if (modelPoints.exits.includes(dataIndex)) {
                                                        return (
                                                            <g>
                                                                <line x1={cx - 3} y1={cy - 3} x2={cx + 3} y2={cy + 3} stroke={colors[index % colors.length]} strokeWidth={2} />
                                                                <line x1={cx - 3} y1={cy + 3} x2={cx + 3} y2={cy - 3} stroke={colors[index % colors.length]} strokeWidth={2} />
                                                            </g>
                                                        );
                                                    }
                                                    return <circle cx={cx} cy={cy} r={0} fill="transparent" />;
                                                }}
                                                activeDot={false}
                                                connectNulls={false}
                                            />
                                        ))}
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    ) : (
                        <div style={{ position: 'relative' }}>
                            {renderTable(mostRecentScores)}
                            {!hasEnoughData && (
                                <div
                                    style={{
                                        position: 'absolute',
                                        top: 0,
                                        left: 0,
                                        right: 0,
                                        bottom: 0,
                                        backgroundColor: 'rgba(0, 0, 0, 0.7)',
                                        backdropFilter: 'blur(8px)',
                                        display: 'flex',
                                        justifyContent: 'center',
                                        alignItems: 'center',
                                        zIndex: 10,
                                    }}
                                >
                                    <div
                                        style={{
                                            color: 'white',
                                            fontSize: '16px',
                                            fontWeight: 'bold',
                                            textAlign: 'center',
                                            padding: '20px',
                                            backgroundColor: 'rgba(0, 0, 0, 0.6)',
                                            borderRadius: '12px',
                                            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                                            backdropFilter: 'blur(4px)',
                                        }}
                                    >
                                        More data needed<br />
                                        <span style={{ fontSize: '14px', fontWeight: 'normal' }}>
                                            {`Unlock with ${MIN_USER_VOTES - userVoteCount} votes. Register a username to see yourself on the user votes leaderboard!`}
                                        </span>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </>
            )}
            <div style={{ marginTop: '20px' }}>
                <h3>Model Selection History</h3>
                <button onClick={toggleSort} style={{
                    backgroundColor: 'var(--vscode-button-background)',
                    color: 'var(--vscode-button-foreground)',
                    border: 'none',
                    padding: '6px 12px',
                    cursor: 'pointer',
                    fontSize: '0.9em',
                    borderRadius: '2px',
                    marginBottom: '10px'
                }}>
                    Sort: Recent {sortAscending ? '↑' : '↓'}
                </button>
                {renderSelectionGroup('Past Day', groupedSelections.pastDay, 'pastDay')}
                {renderSelectionGroup('Past Week', groupedSelections.pastWeek, 'pastWeek')}
                {renderSelectionGroup('Past Month', groupedSelections.pastMonth, 'pastMonth')}
                {renderSelectionGroup('Older', groupedSelections.older, 'older')}
            </div>
            {showRegisterModal && (
                <RegisterModal onRegister={handleRegister} onClose={() => setShowRegisterModal(false)} />
            )}
        </div>
    );
};

const renderApp = () => {
    log('renderApp function called');
    const root = document.getElementById('root');
    if (root) {
        log('Root element found, creating React root');
        const reactRoot = ReactDOM.createRoot(root);
        log('Rendering RankingChart component');
        reactRoot.render(React.createElement(RankingChart));
        vscode.postMessage({ type: 'log', message: 'React component rendered' });
    } else {
        logError('Root element not found');
        vscode.postMessage({ type: 'error', message: 'Root element not found' });
    }
};

// Ensure the DOM is fully loaded before rendering
if (document.readyState === 'loading') {
    log('Document still loading, adding DOMContentLoaded listener');
    document.addEventListener('DOMContentLoaded', renderApp);
} else {
    log('Document already loaded, calling renderApp immediately');
    renderApp();
}

log('RankingChart module loaded');

export default RankingChart;