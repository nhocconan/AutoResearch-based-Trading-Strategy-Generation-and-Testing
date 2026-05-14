# Strategy: 6h_Camarilla_R4S4_1wEMA20_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.229 | +29.4% | -7.8% | 75 | KEEP |
| ETHUSDT | 0.134 | +26.1% | -10.6% | 65 | KEEP |
| SOLUSDT | 0.769 | +84.5% | -15.5% | 53 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.560 | -8.0% | -9.6% | 24 | DISCARD |
| ETHUSDT | 0.155 | +7.5% | -4.8% | 18 | KEEP |
| SOLUSDT | -1.117 | -4.5% | -8.1% | 18 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot R4/S4 breakout with 1w EMA20 trend filter and volume confirmation
# R4/S4 represent stronger breakout levels than R3/S3, reducing false breakouts
# In bull markets: buy when price breaks above R4 with volume spike + price above 1w EMA20
# In bear markets: sell when price breaks below S4 with volume spike + price below 1w EMA20
# Weekly EMA20 provides robust trend filter that adapts to both bull and bear regimes
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "6h_Camarilla_R4S4_1wEMA20_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get 1d data for Camarilla pivot calculation (standard: based on prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: R4 = close + (high-low)*1.1/2, S4 = close - (high-low)*1.1/2
    camarilla_range = (high_1d - low_1d) * 1.1
    r4 = close_1d + camarilla_range / 2
    s4 = close_1d - camarilla_range / 2
    
    # Align Camarilla levels to 6h timeframe (using prior 1d bar's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla breakout signals with 1w trend filter
        # Long: price breaks above R4 + volume spike + price above 1w EMA20
        # Short: price breaks below S4 + volume spike + price below 1w EMA20
        if position == 0:
            if (close[i] > r4_aligned[i] and volume_spike and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < s4_aligned[i] and volume_spike and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S4 (reversal) OR price below 1w EMA20
            if close[i] < s4_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R4 (reversal) OR price above 1w EMA20
            if close[i] > r4_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 00:02
