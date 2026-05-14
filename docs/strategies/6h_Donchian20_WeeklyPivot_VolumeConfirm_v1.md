# Strategy: 6h_Donchian20_WeeklyPivot_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.165 | +12.1% | -12.7% | 155 | FAIL |
| ETHUSDT | 0.298 | +37.5% | -10.4% | 144 | PASS |
| SOLUSDT | 0.725 | +104.0% | -21.5% | 137 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.355 | +11.2% | -8.8% | 47 | PASS |
| SOLUSDT | 0.302 | +10.3% | -8.6% | 49 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation (>1.5x 20-bar MA)
# Donchian breakout captures momentum, weekly pivot filters trend direction (long above weekly pivot, short below),
# volume confirms strength. Works in bull markets via breakouts above weekly pivot and in bear markets via short
# breakdowns below weekly pivot. Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25).

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly pivot points (using prior week's OHLC)
    # Weekly high = max of high over prior 5 trading days (approx 5*24/6 = 20 6h bars, but we use 1d resample logic via prior day's data)
    # Since we have 1d data, we calculate weekly pivot from prior week's 1d OHLC
    # We need to align weekly pivot to 6h bars: each weekly pivot value lasts for 1 week (7*24/6 = 28 6h bars)
    # But we'll use the prior week's OHLC to compute pivot, then align it to 6h bars
    
    # Calculate weekly OHLC from 1d data (assuming 5 trading days per week)
    # We'll use rolling window of 5 days to get weekly high, low, close
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point: (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Donchian(20) channels on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(lookback, 20)  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band, above weekly pivot, and volume confirmation
            if curr_close > highest_high[i-1] and curr_close > weekly_pivot_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, below weekly pivot, and volume confirmation
            elif curr_close < lowest_low[i-1] and curr_close < weekly_pivot_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Donchian lower band or below weekly pivot
            if curr_close < lowest_low[i-1] or curr_close < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price breaking above Donchian upper band or above weekly pivot
            if curr_close > highest_high[i-1] or curr_close > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 16:41
