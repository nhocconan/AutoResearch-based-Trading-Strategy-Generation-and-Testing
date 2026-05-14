# Strategy: 12h_KeltnerBreakout_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.084 | +12.9% | -28.3% | 154 | FAIL |
| ETHUSDT | 0.136 | +26.6% | -18.3% | 191 | PASS |
| SOLUSDT | 1.151 | +238.1% | -21.1% | 145 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.132 | +7.2% | -14.2% | 64 | PASS |
| SOLUSDT | -0.673 | -13.3% | -28.3% | 78 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h_KeltnerBreakout_TrendFilter
12h strategy using Keltner Channel breakout with EMA trend filter.
- Long: Close breaks above Keltner Upper (EMA20 + 2*ATR) + EMA50 > EMA200
- Short: Close breaks below Keltner Lower (EMA20 - 2*ATR) + EMA50 < EMA200
- Exit: Opposite breakout or trend reversal
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Keltner and trend
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA20 (middle of Keltner)
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d ATR (14-period)
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Keltner Bands
    keltner_upper = ema_20_1d + 2 * atr_14
    keltner_lower = ema_20_1d - 2 * atr_14
    
    # Align Keltner bands to 12h
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    
    # 1d EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > keltner_upper_aligned[i]
        breakdown_down = close[i] < keltner_lower_aligned[i]
        
        if position == 0:
            # Long: uptrend + breakout above Keltner Upper
            if uptrend and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + breakdown below Keltner Lower
            elif downtrend and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or breakdown below Keltner Lower
            if not uptrend or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or breakout above Keltner Upper
            if not downtrend or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KeltnerBreakout_TrendFilter"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-18 13:34
