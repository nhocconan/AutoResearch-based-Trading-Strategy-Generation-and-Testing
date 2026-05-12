#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 1-day Camarilla levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    high_d_prev = np.roll(high_d, 1)
    low_d_prev = np.roll(low_d, 1)
    close_d_prev = np.roll(close_d, 1)
    high_d_prev[0] = np.nan
    low_d_prev[0] = np.nan
    close_d_prev[0] = np.nan
    
    # Camarilla calculation
    range_prev = high_d_prev - low_d_prev
    close_prev = close_d_prev
    camarilla_r1 = close_prev + range_prev * 1.1 / 12
    camarilla_s1 = close_prev - range_prev * 1.1 / 12
    
    # 1-day trend (close vs 34 EMA)
    ema34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_d = close_d > ema34_d
    trend_down_d = close_d < ema34_d
    
    # Volume spike detection (volume > 1.5x 20-period MA)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # Align all 1D data to 4H timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    trend_up_d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_d)
    trend_down_d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after EMA warmup
        # Skip if any data is not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(trend_up_d_aligned[i]) or np.isnan(trend_down_d_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: uptrend + price breaks above R1 + volume spike
            if (trend_up_d_aligned[i] and 
                close[i] > camarilla_r1_aligned[i] and 
                volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price breaks below S1 + volume spike
            elif (trend_down_d_aligned[i] and 
                  close[i] < camarilla_s1_aligned[i] and 
                  volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 or trend changes
            if close[i] < camarilla_s1_aligned[i] or not trend_up_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R1 or trend changes
            if close[i] > camarilla_r1_aligned[i] or not trend_down_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals