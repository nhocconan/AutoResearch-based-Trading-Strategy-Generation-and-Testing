# Strategy: 4h_EMA34_12hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.142 | +27.2% | -16.9% | 182 | PASS |
| ETHUSDT | 0.195 | +31.7% | -15.8% | 189 | PASS |
| SOLUSDT | 0.892 | +181.7% | -27.1% | 180 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.581 | -2.2% | -9.3% | 69 | FAIL |
| ETHUSDT | 0.298 | +11.1% | -10.0% | 67 | PASS |
| SOLUSDT | 0.229 | +9.6% | -10.8% | 59 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Trend following with 12h EMA34 trend filter and volume confirmation.
# Long when: Price > 12h EMA34 AND volume > 1.5x 20-period average
# Short when: Price < 12h EMA34 AND volume > 1.5x 20-period average
# Exit when: Price crosses back below/above 12h EMA34
# 12h EMA34 filters direction, volume confirms strength, 4h price action triggers entries.
# Works in trending markets by capturing sustained moves. Target: 20-30 trades/year per symbol.
name = "4h_EMA34_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h EMA34 ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_12h_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price > 12h EMA34 + volume spike
            if price > ema34 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: Price < 12h EMA34 + volume spike
            elif price < ema34 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below 12h EMA34
            if price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above 12h EMA34
            if price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 01:47
