# Strategy: 4h_1d_Keltner_Channel_Breakout_With_Volume_and_Trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.127 | +7.9% | -19.6% | 81 | FAIL |
| ETHUSDT | 0.462 | +61.9% | -15.5% | 74 | PASS |
| SOLUSDT | 0.461 | +77.5% | -46.2% | 76 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.539 | +17.2% | -12.8% | 26 | PASS |
| SOLUSDT | -0.812 | -15.3% | -23.8% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_1d_Keltner_Channel_Breakout_With_Volume_and_Trend_v1
Hypothesis: Keltner Channel (ATR-based) breakouts with volume expansion and trend filter capture institutional moves.
Uses 4h price breaking above/below 2xATR Keltner bands with volume > 1.5x 20-period average and EMA50 trend filter.
Works in both bull and bear markets by trading breakouts in direction of trend. Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for Keltner Channels
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Keltner Channel bands (2x ATR)
    upper_keltner = ema50 + (2 * atr)
    lower_keltner = ema50 - (2 * atr)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(ema50[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above upper Keltner with volume expansion and uptrend
        long_breakout = (close[i] > upper_keltner[i] and 
                        volume_expansion[i] and 
                        close[i] > ema50[i])
        
        # Short breakout: price breaks below lower Keltner with volume expansion and downtrend
        short_breakout = (close[i] < lower_keltner[i] and 
                         volume_expansion[i] and 
                         close[i] < ema50[i])
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Keltner_Channel_Breakout_With_Volume_and_Trend_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 19:29
