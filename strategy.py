#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with 1d EMA trend filter and volume spike confirmation. In bullish 1d trend, long on break above R1; in bearish 1d trend, short on break below S1. Volume must be > 1.5x 20-period average to confirm institutional interest. Uses discrete 0.25 position sizing to limit fee drag. Target: 50-150 total trades over 4 years = 12-37/year.
"""

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
    
    # Get 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for Camarilla levels and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    prev_close_12h = df_12h['close'].shift(1).values
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    camarilla_r1 = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 12
    camarilla_s1 = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Calculate volume confirmation: current 12h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(34) and volume MA(20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend using EMA34
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Breakout entries: follow 1d trend direction
            # In bullish 1d trend: long on break above R1
            # In bearish 1d trend: short on break below S1
            long_setup = (close[i] > camarilla_r1_aligned[i]) and htf_1d_bullish and volume_spike[i]
            short_setup = (close[i] < camarilla_s1_aligned[i]) and htf_1d_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price returns to Camarilla H5 level or trend reverses
            camarilla_h5 = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 * 2 / 12  # H5 = close + 1.1*(HL)*2/12
            camarilla_h5_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h5)
            exit_signal = (close[i] < camarilla_h5_aligned[i]) or (not htf_1d_bullish)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price returns to Camarilla L5 level or trend reverses
            camarilla_l5 = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 * 2 / 12  # L5 = close - 1.1*(HL)*2/12
            camarilla_l5_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l5)
            exit_signal = (close[i] > camarilla_l5_aligned[i]) or htf_1d_bullish
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0