# BTC bot

- Source URL: https://www.tradingview.com/script/DPBeZbMP-BTC-bot/
- Pine file: `raw-pine/bulk/DPBeZbMP-BTC-bot.pine`
- Classification: `unsupported`
- Timeframe: `15m`
- Attempts used: `1`
- Result: `unsupported`
- Reason: Uses security() with lookahead=true causing repainting; strategy.exit requires next-bar approximation

## Adaptations

- Remove lookahead=true from security calls
- Convert strategy.exit to next-bar signal changes
- Replace strategy.risk with position sizing logic
