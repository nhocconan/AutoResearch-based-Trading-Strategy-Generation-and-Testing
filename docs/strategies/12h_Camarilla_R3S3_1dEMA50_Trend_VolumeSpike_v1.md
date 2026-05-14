# Strategy: 12h_Camarilla_R3S3_1dEMA50_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.131 | +25.4% | -7.4% | 82 | PASS |
| ETHUSDT | 0.090 | +24.1% | -9.2% | 70 | PASS |
| SOLUSDT | 0.413 | +51.0% | -16.6% | 70 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.010 | -2.1% | -6.4% | 33 | FAIL |
| ETHUSDT | 0.310 | +9.5% | -3.9% | 27 | PASS |
| SOLUSDT | -0.880 | -5.0% | -14.6% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level AND price > 1d EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 level AND price < 1d EMA50 AND volume > 2.0x 20-bar average.
# Exit when price crosses the Camarilla pivot point (midpoint of the day).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Camarilla levels from 1d provide robust intraday support/resistance that works in both trending and ranging markets,
# while 1d EMA50 filters for the dominant trend to avoid counter-trend entries.
# Volume spike confirms institutional participation in breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Camarilla_R3S3_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla levels (R3, S3, pivot) from previous completed day to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Shift by 1 to use only completed 1d bars
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    close_1d_shifted = np.roll(close_1d_arr, 1)
    high_1d_shifted[0] = np.nan
    low_1d_shifted[0] = np.nan
    close_1d_shifted[0] = np.nan
    
    # Calculate pivot and ranges from previous completed day
    pivot = (high_1d_shifted + low_1d_shifted + close_1d_shifted) / 3.0
    range_1d = high_1d_shifted - low_1d_shifted
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S3 = close - (range * 1.1/4)
    camarilla_r3 = close_1d_shifted + (range_1d * 1.1 / 4)
    camarilla_s3 = close_1d_shifted - (range_1d * 1.1 / 4)
    camarilla_pivot = pivot  # Pivot point as exit level
    
    # Align 1d indicators to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(51, 20)  # warmup for EMA and Camarilla calculations
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R3, uptrend (price > 1d EMA50), volume confirmation
            if (curr_high > camarilla_r3_aligned[i] and 
                curr_close > ema_50_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S3, downtrend (price < 1d EMA50), volume confirmation
            elif (curr_low < camarilla_s3_aligned[i] and 
                  curr_close < ema_50_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below Camarilla pivot
            if curr_close < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above Camarilla pivot
            if curr_close > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-30 15:40
