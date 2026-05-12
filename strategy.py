#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

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
    open_time = prices['open_time']
    
    # === Pre-compute session filter (08-20 UTC) ===
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 4h Trend Filter (HTF) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # === 1d Volume Filter (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # === 1h Camarilla Levels ===
    high_1h = high
    low_1h = low
    close_1h = close
    
    # Calculate previous day's Camarilla levels
    prev_high = pd.Series(high_1h).rolling(window=24, min_periods=24).max().shift(24).values
    prev_low = pd.Series(low_1h).rolling(window=24, min_periods=24).min().shift(24).values
    prev_close = pd.Series(close_1h).rolling(window=24, min_periods=24).mean().shift(24).values
    
    range_ = prev_high - prev_low
    R4 = prev_close + range_ * 1.1 / 2
    R3 = prev_close + range_ * 1.1 / 4
    S3 = prev_close - range_ * 1.1 / 4
    S4 = prev_close - range_ * 1.1 / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(R4[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(S4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade during session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > R4 + 4h Uptrend + 1d Volume Spike
            if (close[i] > R4[i] and 
                close[i] > ema20_4h_aligned[i] and 
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # Short: Close < S4 + 4h Downtrend + 1d Volume Spike
            elif (close[i] < S4[i] and 
                  close[i] < ema20_4h_aligned[i] and 
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Close < R3 OR 4h trend turns down
            if close[i] < R3[i] or close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Close > S3 OR 4h trend turns up
            if close[i] > S3[i] or close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals