# Strategy: 6H_WILLIAMS_FRACTAL_REVERSAL_1D_VOLUME_FILTER

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.091 | +24.1% | -14.0% | 83 | PASS |
| ETHUSDT | 0.680 | +62.1% | -8.9% | 70 | PASS |
| SOLUSDT | 1.492 | +282.9% | -19.0% | 77 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.125 | -2.3% | -6.7% | 31 | FAIL |
| ETHUSDT | 0.079 | +6.6% | -8.8% | 26 | PASS |
| SOLUSDT | -0.129 | +3.7% | -8.5% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6H_WILLIAMS_FRACTAL_REVERSAL_1D_VOLUME_FILTER
Hypothesis: Daily Williams Fractals identify reversal points. Price rejection at
bearish fractal (sell signal) or bullish fractal (buy signal) with volume
confirmation and EMA21 trend filter works in both bull (buy dips) and bear
(sell rallies) markets. Uses 6h timeframe for entries with 1d fractals as
structure. Target: 20-50 trades/year on 6h timeframe (80-200 total over 4 years).
"""
name = "6H_WILLIAMS_FRACTAL_REVERSAL_1D_VOLUME_FILTER"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Williams Fractals and filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals (need 5 bars: 2 left, center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # EMA21 for trend filter
    ema21 = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume spike: current 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    # Williams Fractals need 2-bar confirmation after the center bar
    bearish_fractal_confirm = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirm = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema21_aligned = align_htf_to_ltf(prices, df_1d, ema21)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 2  # Need fractal formation
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(bearish_fractal_confirm[i]) or np.isnan(bullish_fractal_confirm[i]) or 
            np.isnan(ema21_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price holds above bullish fractal with volume spike in uptrend
            if (low[i] > bullish_fractal_confirm[i] and 
                volume_spike[i] and 
                close[i] > ema21_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price holds below bearish fractal with volume spike in downtrend
            elif (high[i] < bearish_fractal_confirm[i] and 
                  volume_spike[i] and 
                  close[i] < ema21_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below bullish fractal or trend reversal
            if (low[i] <= bullish_fractal_confirm[i] or 
                close[i] < ema21_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above bearish fractal or trend reversal
            if (high[i] >= bearish_fractal_confirm[i] or 
                close[i] > ema21_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 10:11
