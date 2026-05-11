# 4h_Camarilla_R1S1_Breakout_TrendFilter_Volume_1d
# Hypothesis: Use daily pivot levels (from 1d data) for more reliable support/resistance.
# Breakouts from daily S1/R1 with volume and trend confirmation (4h EMA50).
# Daily pivots reduce noise and provide stronger levels than 4h pivots.
# This should work in both bull and bear markets by following the intermediate trend.
# Target: 20-40 trades/year to avoid fee drag.

name = "4h_Camarilla_R1S1_Breakout_TrendFilter_Volume_1d"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Data for Daily Pivots ---
    df_1d = get_htf_data(prices, '1d')
    # Calculate daily pivot from previous day's OHLC
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Shift to get previous day's values (today's pivot uses yesterday's data)
    prev_high_1d = np.roll(prev_high_1d, 1)
    prev_low_1d = np.roll(prev_low_1d, 1)
    prev_close_1d = np.roll(prev_close_1d, 1)
    # First day will have NaN
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    
    # Daily Camarilla levels
    R1_1d = pivot_1d + (range_1d * 1.1 / 12)
    S1_1d = pivot_1d - (range_1d * 1.1 / 12)
    
    # Align daily levels to 4h timeframe (wait for daily close)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # --- 4h EMA50 Trend Filter ---
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA50 and pivot calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: price breaks above daily R1 with volume, above EMA50
            if (close[i] > R1_1d_aligned[i] and 
                volume_spike and 
                close[i] > ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S1 with volume, below EMA50
            elif (close[i] < S1_1d_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_50[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of momentum
            if position == 1:
                # Exit long: price breaks below daily S1 (reversal signal)
                if close[i] < S1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above daily R1 (reversal signal)
                if close[i] > R1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals