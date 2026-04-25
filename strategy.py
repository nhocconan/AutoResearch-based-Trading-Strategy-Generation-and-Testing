#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Daily Camarilla R1/S1 breakout with 1-week EMA trend filter and volume confirmation. Uses HTF 1w for trend alignment (price > 1w EMA for long, < 1w EMA for short) to reduce whipsaw. Volume confirmation requires >2.0x 20-bar mean volume. Targets 15-25 trades/year per symbol by requiring strong volume spike and clear weekly trend. Designed to work in both bull (breakouts with volume) and bear (trend-following shorts) markets via disciplined entry/exit on 1d timeframe.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    # Need to get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12
    camarilla_r1 = close_1d + (1.1 * (high_1d - low_1d) / 12)
    camarilla_s1 = close_1d - (1.1 * (high_1d - low_1d) / 12)
    
    # Align Camarilla levels to 1d timeframe (use previous bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in uptrend (price > 1w EMA) with volume confirmation
            # Short: price breaks below Camarilla S1 in downtrend (price < 1w EMA) with volume confirmation
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema_34_1w_aligned[i]) and vol_confirm[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema_34_1w_aligned[i]) and vol_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Camarilla S1 (mean reversion)
            exit_signal = close[i] < camarilla_s1_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla R1 (mean reversion)
            exit_signal = close[i] > camarilla_r1_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0