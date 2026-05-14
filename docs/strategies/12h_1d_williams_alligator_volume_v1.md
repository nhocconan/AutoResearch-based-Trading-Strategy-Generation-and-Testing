# Strategy: 12h_1d_williams_alligator_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.017 | -18.8% | -25.9% | 91 | DISCARD |
| ETHUSDT | -0.286 | +3.5% | -20.8% | 86 | DISCARD |
| SOLUSDT | 0.987 | +148.8% | -17.5% | 63 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.281 | +10.3% | -9.6% | 23 | KEEP |

## Code
```python
#!/usr/bin/env python3
# 12h_1d_williams_alligator_volume_v1
# Hypothesis: Use Williams Alligator on 1d to determine trend direction (jaw/teeth/lips alignment) and 12h for entry timing with volume confirmation.
# Long when Alligator is bullish (lips > teeth > jaw) and price crosses above 12h teeth with volume.
# Short when Alligator is bearish (lips < teeth < jaw) and price crosses below 12h teeth with volume.
# Williams Alligator uses SMAs with specific offsets to avoid look-ahead.
# Designed to work in both bull and bear markets by following the Alligator's trend direction.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) by requiring Alligator alignment and volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_williams_alligator_volume_v1"
timeframe = "12h"
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
    
    # Get 1d data for Williams Alligator (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Get 12h data for entry timing
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    close_1d = df_1d['close'].values
    
    # Calculate SMMA (Smoothed Moving Average) - equivalent to Wilder's smoothing
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Apply shifts (jaw: 8, teeth: 5, lips: 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Invalidate the shifted values (set to NaN for the shifted periods)
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 12h SMMA (8-period) for entry timing (teeth equivalent)
    close_12h = df_12h['close'].values
    def smma_12h(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    teeth_12h_raw = smma_12h(close_12h, 8)
    teeth_12h = np.roll(teeth_12h_raw, 5)  # Shift 5 bars
    teeth_12h[:5] = np.nan
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    
    # Volume confirmation: volume > 1.3x average of last 24 periods (2 days in 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(teeth_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Determine Alligator alignment
        bullish_alligator = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_alligator = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 12h teeth or Alligator turns bearish
            if close[i] < teeth_12h_aligned[i] or bearish_alligator:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above 12h teeth or Alligator turns bullish
            if close[i] > teeth_12h_aligned[i] or bullish_alligator:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: Alligator bullish and price crosses above 12h teeth with volume
            if bullish_alligator and close[i] > teeth_12h_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Alligator bearish and price crosses below 12h teeth with volume
            elif bearish_alligator and close[i] < teeth_12h_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-11 12:22
