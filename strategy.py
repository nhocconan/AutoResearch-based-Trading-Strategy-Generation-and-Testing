#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike
Hypothesis: On 1h timeframe, use Camarilla R1/S1 breakouts for entry timing, with 4h EMA50 trend filter and 1d volume spike confirmation (>2x 20-median). This uses higher timeframes for signal direction (4h trend, 1d volume regime) and 1h only for precise entry timing, reducing whipsaw. Discrete sizing (0.20) limits fee drag. Target: 60-150 trades over 4 years. Works in bull via breakout continuation and in bear by avoiding low-volume counter-trend trades via 1d volume filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 1h (based on previous bar's range)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    
    # Volume confirmation: 1d volume > 2x 20-period median (robust to outliers)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_series = pd.Series(vol_1d)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike_1d = vol_1d > (vol_median * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Start after warmup (need 20-period for volume median, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R1 with 4h uptrend and 1d volume spike
        long_condition = (close[i] > r1[i]) and (close[i] > ema_50_4h_aligned[i]) and volume_spike_1d_aligned[i]
        # Short logic: break below S1 with 4h downtrend and 1d volume spike
        short_condition = (close[i] < s1[i]) and (close[i] < ema_50_4h_aligned[i]) and volume_spike_1d_aligned[i]
        
        # Exit logic: opposite Camarilla level (S1/R1) or trend reversal
        exit_long = (close[i] < s1[i]) or (close[i] < ema_50_4h_aligned[i])
        exit_short = (close[i] > r1[i]) or (close[i] > ema_50_4h_aligned[i])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0