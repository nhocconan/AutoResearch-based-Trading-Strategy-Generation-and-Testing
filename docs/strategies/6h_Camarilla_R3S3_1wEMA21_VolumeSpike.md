# Strategy: 6h_Camarilla_R3S3_1wEMA21_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.578 | +43.1% | -6.5% | 110 | KEEP |
| ETHUSDT | 0.374 | +36.5% | -11.8% | 98 | KEEP |
| SOLUSDT | 0.905 | +103.4% | -13.8% | 90 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.975 | -0.2% | -8.4% | 34 | DISCARD |
| ETHUSDT | 0.826 | +15.2% | -10.2% | 27 | KEEP |
| SOLUSDT | -0.401 | +0.7% | -11.3% | 34 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with weekly trend filter and volume confirmation
# Uses weekly EMA(21) to capture major trend direction and avoid counter-trend trades
# Camarilla levels from daily timeframe provide institutional-grade breakout levels
# Volume confirmation (>1.8x 20 EMA volume) filters false breakouts in choppy markets
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in bull markets (continuation at R3/R4 with weekly uptrend) and bear markets (continuation at S3/S4 with weekly downtrend)
# Weekly trend filter ensures alignment with major market structure, reducing whipsaws

name = "6h_Camarilla_R3S3_1wEMA21_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for prior completed bar
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior completed 1d Camarilla levels (R3, R4, S3, S4)
    daily_range = high_1d - low_1d
    camarilla_r4 = close_1d + daily_range * 1.1 / 2
    camarilla_r3 = close_1d + daily_range * 1.1 / 4
    camarilla_s3 = close_1d - daily_range * 1.1 / 4
    camarilla_s4 = close_1d - daily_range * 1.1 / 2
    
    # Shift to use prior completed 1d bar (avoid look-ahead)
    camarilla_r4_shifted = np.roll(camarilla_r4, 1)
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_s4_shifted = np.roll(camarilla_s4, 1)
    camarilla_r4_shifted[0] = np.nan
    camarilla_r3_shifted[0] = np.nan
    camarilla_s3_shifted[0] = np.nan
    camarilla_s4_shifted[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_shifted)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_shifted)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_shifted)
    
    # Get 1w data for weekly EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:  # Need enough data for EMA21 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(21) trend filter from prior completed weekly bar
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_shifted = np.roll(ema_21_1w, 1)
    ema_21_1w_shifted[0] = np.nan
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND price > weekly EMA21 AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_21_1w_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND price < weekly EMA21 AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_21_1w_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla S3 OR price crosses below weekly EMA21
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla R3 OR price crosses above weekly EMA21
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 12:02
