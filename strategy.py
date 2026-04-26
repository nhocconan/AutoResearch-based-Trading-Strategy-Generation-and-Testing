#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_HTFVolumeSpike_v1
Hypothesis: Trade 12h breakouts from Camarilla R1/S1 levels with 1w EMA50 trend filter and 1d volume confirmation.
Works in both bull/bear via 1w trend filter; Camarilla levels provide structure in ranging markets.
Uses discrete size 0.25 to limit fee drag. Target 12-37 trades/year on 12h timeframe.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot levels and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Camarilla pivot levels (R1, S1) from previous 1d bar
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    
    # Calculate Camarilla R1 and S1 for previous 1d bar
    # Formula: R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 12
    
    # 1d volume confirmation: volume > 2.0x 20-period average
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_20 = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d / np.maximum(volume_ma_20, 1e-10) > 2.0
    
    # Align all HTF data to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
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
            # Long: price breaks above R1 + volume spike + 1w uptrend
            long_breakout = close[i] > camarilla_r1_aligned[i]
            long_signal = long_breakout and volume_spike_1d_aligned[i] and trend_1w_uptrend
            
            # Short: price breaks below S1 + volume spike + 1w downtrend
            short_breakout = close[i] < camarilla_s1_aligned[i]
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