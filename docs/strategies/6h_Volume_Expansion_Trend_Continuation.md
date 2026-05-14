# Strategy: 6h_Volume_Expansion_Trend_Continuation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.262 | +35.5% | -16.7% | 69 | PASS |
| ETHUSDT | 0.192 | +31.3% | -17.0% | 71 | PASS |
| SOLUSDT | 0.774 | +139.9% | -33.6% | 73 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.415 | -0.1% | -8.6% | 31 | FAIL |
| ETHUSDT | 0.714 | +21.4% | -9.4% | 20 | PASS |
| SOLUSDT | 0.284 | +11.0% | -12.5% | 21 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "6h_Volume_Expansion_Trend_Continuation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily trend filter: EMA34 on 1d close
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume expansion: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_expansion = volume > (2.0 * vol_ma)
    
    # Price momentum: close > open (bullish candle) or close < open (bearish candle)
    bullish_candle = close > open_
    bearish_candle = close < open_
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # ensure EMA34 has enough data
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above daily EMA34 + bullish candle + volume expansion
            if (close[i] > ema34_1d_aligned[i]) and bullish_candle[i] and vol_expansion[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34 + bearish candle + volume expansion
            elif (close[i] < ema34_1d_aligned[i]) and bearish_candle[i] and vol_expansion[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below daily EMA34
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above daily EMA34
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 00:54
