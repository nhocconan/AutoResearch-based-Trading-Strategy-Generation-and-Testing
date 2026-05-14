# Strategy: 6h_Donchian20_Breakout_1dWeeklyPivot_Direction_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.002 | +19.7% | -12.0% | 135 | PASS |
| ETHUSDT | 0.250 | +34.3% | -12.4% | 121 | PASS |
| SOLUSDT | 0.592 | +85.2% | -29.9% | 108 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.936 | -4.4% | -9.5% | 51 | FAIL |
| ETHUSDT | 0.088 | +6.7% | -8.0% | 43 | PASS |
| SOLUSDT | -0.761 | -8.4% | -18.5% | 37 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
Hypothesis: Donchian breakouts capture momentum. Weekly pivot (from prior week) provides
institutional bias: long only when price above weekly pivot, short only when below.
Volume spike confirms institutional participation. Works in bull markets (buy breakouts
in uptrend bias) and bear markets (sell breakdowns in downtrend bias).
6h timeframe targets 12-37 trades/year (50-150 over 4 years).
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
    
    # Load 1d data ONCE before loop for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week's OHLC (using daily data)
    # Need to resample daily to weekly manually since we only have daily HTF
    # But we can approximate: use prior week's high/low/close from daily
    # For simplicity, use prior day's OHLC as proxy for weekly (more frequent updates)
    # Better: calculate true weekly by grouping df_1d into weeks
    # We'll do: weekly high = max of last 7 daily highs, etc.
    weekly_high = pd.Series(df_1d['high']).rolling(window=7, min_periods=7).max().shift(1).values
    weekly_low = pd.Series(df_1d['low']).rolling(window=7, min_periods=7).min().shift(1).values
    weekly_close = pd.Series(df_1d['close']).rolling(window=7, min_periods=7).last().shift(1).values
    
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 20)  # Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Bias filter: price relative to weekly pivot
        bullish_bias = curr_close > weekly_pivot_aligned[i]
        bearish_bias = curr_close < weekly_pivot_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND bullish bias AND volume spike
            long_entry = (curr_high > donchian_high[i]) and bullish_bias and vol_spike
            # Short: price breaks below Donchian low AND bearish bias AND volume spike
            short_entry = (curr_low < donchian_low[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian low (breakdown) OR loss of bullish bias
            if (curr_low < donchian_low[i]) or (curr_close < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian high (breakout) OR loss of bearish bias
            if (curr_high > donchian_high[i]) or (curr_close > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dWeeklyPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 06:48
