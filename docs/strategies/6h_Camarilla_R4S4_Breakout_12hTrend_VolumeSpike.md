# Strategy: 6h_Camarilla_R4S4_Breakout_12hTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.678 | +54.4% | -7.5% | 175 | PASS |
| ETHUSDT | 0.257 | +34.0% | -10.0% | 162 | PASS |
| SOLUSDT | 0.629 | +79.7% | -15.9% | 138 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.997 | -3.7% | -8.4% | 61 | FAIL |
| ETHUSDT | 1.165 | +26.6% | -6.5% | 52 | PASS |
| SOLUSDT | -0.020 | +5.0% | -11.5% | 46 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_12hTrend_VolumeSpike
Hypothesis: On 6h timeframe, Camarilla R4/S4 breakouts with 12h EMA50 trend filter and volume spike (>2.0x 20-bar avg) captures strong institutional moves. R4/S4 levels represent stronger breakout points than R1/S1, reducing false signals. Volume spike confirms participation. Trend filter ensures alignment with higher timeframe momentum. Designed for low trade frequency (12-30/year) to minimize fee drag in 6h timeframe. Works in both bull and bear markets via trend filter.
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
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h bar (R4, S4)
    # Camarilla: R4 = close + 1.1*(high-low)/2, S4 = close - 1.1*(high-low)/2
    # Use previous completed 12h bar to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close_12h[:-1]])
    prev_high = np.concatenate([[np.nan], high_12h[:-1]])
    prev_low = np.concatenate([[np.nan], low_12h[:-1]])
    
    camarilla_range = prev_high - prev_low
    r4 = prev_close + 1.1 * camarilla_range / 2
    s4 = prev_close - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20)  # EMA50, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_50_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Camarilla R4/S4 breakout with trend and volume
            # Long: price breaks above R4 with uptrend (close > EMA50) and volume spike
            long_signal = (high_val > r4_val) and (close_val > ema_val) and volume_spike
            # Short: price breaks below S4 with downtrend (close < EMA50) and volume spike
            short_signal = (low_val < s4_val) and (close_val < ema_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S4 (exit long)
            if close_val < s4_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R4 (exit short)
            if close_val > r4_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 23:29
