#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_HTFVolumeSpike_v1
Hypothesis: Trade 12h breakouts from Camarilla R1/S1 levels with 1w EMA50 trend filter and 1d volume spike confirmation. Uses weekly trend for major market direction (works in bull/bear) and daily volume spikes for institutional interest confirmation. Discrete size 0.25 limits fee drag. Target 12-30 trades/year.
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
    
    # Get 1d data for volume confirmation (more reliable than 12h volume)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d volume spike: volume > 2.0x 20-period average
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d / np.maximum(volume_ma_1d, 1e-10) > 2.0
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1w Camarilla pivot levels (R1, S1) from previous 1w bar
    # Formula: R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan  # first bar has no previous
    
    camarilla_r1_1w = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) * 1.1 / 12
    camarilla_s1_1w = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) * 1.1 / 12
    
    # 1w EMA50 trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1w data to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1_1w)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), 1d volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend alignment
        trend_1w_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 1d volume spike + 1w uptrend
            long_breakout = (close[i] > camarilla_r1_aligned[i]) and \
                           (close[i-1] <= camarilla_r1_aligned[i-1])
            long_signal = long_breakout and volume_spike_1d_aligned[i] and trend_1w_uptrend
            
            # Short: price breaks below S1 + 1d volume spike + 1w downtrend
            short_breakout = (close[i] < camarilla_s1_aligned[i]) and \
                           (close[i-1] >= camarilla_s1_aligned[i-1])
            short_signal = short_breakout and volume_spike_1d_aligned[i] and trend_1w_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches S1 level OR 1w trend turns down
            if (close[i] < camarilla_s1_aligned[i] or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches R1 level OR 1w trend turns up
            if (close[i] > camarilla_r1_aligned[i] or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_HTFVolumeSpike_v1"
timeframe = "12h"
leverage = 1.0