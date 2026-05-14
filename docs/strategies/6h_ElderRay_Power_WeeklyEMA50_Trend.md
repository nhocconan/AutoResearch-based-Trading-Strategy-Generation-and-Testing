# Strategy: 6h_ElderRay_Power_WeeklyEMA50_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.030 | +21.4% | -11.7% | 60 | PASS |
| ETHUSDT | 0.121 | +25.7% | -12.3% | 72 | PASS |
| SOLUSDT | 1.102 | +176.2% | -27.4% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.866 | -0.1% | -6.8% | 19 | FAIL |
| ETHUSDT | 0.383 | +10.6% | -9.3% | 15 | PASS |
| SOLUSDT | -0.291 | +1.7% | -6.6% | 18 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power with 1d Weekly Trend Filter.
Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) AND price > 1d EMA50 (weekly uptrend).
Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) AND price < 1d EMA50 (weekly downtrend).
Exit when Elder Ray powers converge (|Bull Power| < |Bear Power| for longs, |Bear Power| < |Bull Power| for shorts) 
or weekly trend reverses.
Uses 1d for EMA50 trend filter, 6h for Elder Ray calculation.
Target: 50-150 total trades over 4 years (12-37/year). Elder Ray captures momentum strength, 
weekly EMA50 filters for higher-timeframe trend alignment to reduce false signals in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Elder Ray Power on 6h timeframe
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        bp = bull_power[i]
        br = bear_power[i]
        price = close[i]
        ema50 = ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying) AND Bear Power < 0 (weak selling) 
            #        AND price > 1d EMA50 (weekly uptrend)
            if bp > 0 and br < 0 and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 (strong selling) AND Bull Power < 0 (weak buying)
            #        AND price < 1d EMA50 (weekly downtrend)
            elif br > 0 and bp < 0 and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= Bear Power (momentum weakening) OR price < 1d EMA50 (trend reversal)
            if bp <= br or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power <= Bull Power (momentum weakening) OR price > 1d EMA50 (trend reversal)
            if br <= bp or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_WeeklyEMA50_Trend"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 19:27
