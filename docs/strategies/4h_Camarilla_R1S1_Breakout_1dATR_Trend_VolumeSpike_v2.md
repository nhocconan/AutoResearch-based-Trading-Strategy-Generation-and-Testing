# Strategy: 4h_Camarilla_R1S1_Breakout_1dATR_Trend_VolumeSpike_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.197 | +29.6% | -9.9% | 181 | PASS |
| ETHUSDT | 0.168 | +28.7% | -14.1% | 170 | PASS |
| SOLUSDT | 0.713 | +101.4% | -24.0% | 161 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.434 | +1.3% | -5.6% | 66 | FAIL |
| ETHUSDT | 0.890 | +21.2% | -10.7% | 65 | PASS |
| SOLUSDT | 0.217 | +8.9% | -8.7% | 54 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dATR_Trend_VolumeSpike_v2
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 1d ATR-based trend filter (price > SMA50 + ATR or < SMA50 - ATR) and volume confirmation (2.0x median). Only trade in direction of 1d ATR-adjusted trend to reduce whipsaws. Uses ATR trailing stop (2.5x). Target: 30-60 trades/year on 4h. Works in bull/bear by adapting trend filter to volatility.
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
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d SMA(50) and ATR(14) for trend filter
    sma_50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'], 
                       np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)),
                                  np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_r1 = prev_close_1d + 1.000/6 * (prev_high_1d - prev_low_1d)
    camarilla_s1 = prev_close_1d - 1.000/6 * (prev_high_1d - prev_low_1d)
    camarilla_r2 = prev_close_1d + 2.000/6 * (prev_high_1d - prev_low_1d)
    camarilla_s2 = prev_close_1d - 2.000/6 * (prev_high_1d - prev_low_1d)
    camarilla_r3 = prev_close_1d + 3.000/6 * (prev_high_1d - prev_low_1d)
    camarilla_s3 = prev_close_1d - 3.000/6 * (prev_high_1d - prev_low_1d)
    camarilla_r4 = prev_close_1d + 4.000/6 * (prev_high_1d - prev_low_1d)
    camarilla_s4 = prev_close_1d - 4.000/6 * (prev_high_1d - prev_low_1d)
    
    # Align HTF indicators to 4h timeframe
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 2.0x median volume (20-period) for signal
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops (4h ATR)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of SMA(50), ATR(14) 1d, volume median (20), ATR (14) 4h
    start_idx = max(50, 14, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        sma_50_1d_val = sma_50_1d_aligned[i]
        atr_14_1d_val = atr_14_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        
        # Trend filter: price > SMA50 + 0.5*ATR (uptrend) or < SMA50 - 0.5*ATR (downtrend)
        uptrend = close_val > (sma_50_1d_val + 0.5 * atr_14_1d_val)
        downtrend = close_val < (sma_50_1d_val - 0.5 * atr_14_1d_val)
        
        if position == 0:
            # Long: break above R1 with volume spike, and uptrend
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          uptrend
            
            # Short: break below S1 with volume spike, and downtrend
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dATR_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-26 02:25
