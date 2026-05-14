# Strategy: 4h_Williams_Alligator_1dEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.070 | +17.3% | -9.6% | 320 | FAIL |
| ETHUSDT | 0.004 | +19.4% | -11.1% | 291 | PASS |
| SOLUSDT | 0.311 | +42.8% | -20.6% | 220 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.279 | +9.7% | -8.1% | 105 | PASS |
| SOLUSDT | -0.924 | -7.9% | -17.6% | 81 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h Williams Alligator with 1d EMA50 Trend Filter and Volume Spike Confirmation
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trending vs ranging markets. 
In trending markets (Alligator awake), we trade breakouts in the direction of the 1d EMA50 trend.
In ranging markets (Alligator sleeping), we fade moves at extremes. Volume spike (>2.0x 20-bar vol MA) confirms momentum.
Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets by adapting to regime.
Target: 20-40 trades/year to avoid fee drag while capturing strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 4h data
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(src, length):
        """Smoothed Moving Average"""
        result = np.full_like(src, np.nan, dtype=float)
        if len(src) < length:
            return result
        # First value is SMA
        result[length-1] = np.mean(src[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CLOSE) / length
        for i in range(length, len(src)):
            result[i] = (result[i-1] * (length-1) + src[i]) / length
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines (Jaw: 8, Teeth: 5, Lips: 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set shifted values to NaN for invalid positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator, EMA50, and volume MA
    start_idx = max(51, 20)  # 51 for EMA50 (50 + 1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(lips_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or 
            np.isnan(jaw_shifted[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        vol_ma = vol_ma_20[i]
        
        # Alligator conditions
        alligator_awake = (lips_val > teeth_val > jaw_val) or (lips_val < teeth_val < jaw_val)
        alligator_sleeping = not alligator_awake
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            if alligator_awake:
                # Trending market: trade in direction of 1d EMA50
                # Long: price above EMA50 + lips above teeth (bullish alignment) + volume confirmation
                long_signal = price_above_ema and (lips_val > teeth_val) and volume_confirm
                # Short: price below EMA50 + lips below teeth (bearish alignment) + volume confirmation
                short_signal = price_below_ema and (lips_val < teeth_val) and volume_confirm
            else:
                # Ranging market: fade extremes
                # Long: price near lips (support) + volume confirmation
                # Short: price near teeth/jaw (resistance) + volume confirmation
                long_signal = (curr_close <= lips_val * 1.005) and volume_confirm  # near lips
                short_signal = (curr_close >= jaw_val * 0.995) and volume_confirm   # near jaw
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator sleeping OR price crosses below teeth
            if alligator_sleeping or (curr_close < teeth_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator sleeping OR price crosses above teeth
            if alligator_sleeping or (curr_close > teeth_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 03:28
