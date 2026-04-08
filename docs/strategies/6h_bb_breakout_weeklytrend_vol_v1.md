# Strategy: 6h_bb_breakout_weeklytrend_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.154 | +15.2% | -9.4% | 72 | DISCARD |
| ETHUSDT | 0.300 | +33.7% | -8.9% | 55 | KEEP |
| SOLUSDT | 0.799 | +89.3% | -19.2% | 49 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.578 | +11.9% | -5.7% | 17 | KEEP |
| SOLUSDT | -1.467 | -7.5% | -16.6% | 13 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with weekly trend filter and volume confirmation
# Enter long when price breaks above upper BB(20,2) with close > weekly EMA(50) and volume > 2x avg
# Enter short when price breaks below lower BB(20,2) with close < weekly EMA(50) and volume > 2x avg
# Exit when price returns to middle BB or opposite band is touched
# Uses weekly trend to filter breakouts, targeting 50-150 total trades over 4 years
# Bollinger Bands capture volatility expansion, weekly EMA ensures trend alignment

name = "6h_bb_breakout_weeklytrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_middle = bb_middle.values
    bb_upper = bb_upper.values
    bb_lower = bb_lower.values
    
    # Weekly EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price touches middle BB OR touches lower BB (reversal)
            if close[i] <= bb_middle[i] or close[i] <= bb_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches middle BB OR touches upper BB (reversal)
            if close[i] >= bb_middle[i] or close[i] >= bb_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts: price outside BB + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > bb_upper[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above upper BB with weekly uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < bb_lower[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakout below lower BB with weekly downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals
```

## Last Updated
2026-04-07 04:13
