#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v2
Hypothesis: Camarilla R1/S1 breakouts on 12h with 1w EMA50 trend filter and volume spike confirmation (>1.8x 20-bar avg). Uses discrete sizing (0.30) to limit trades (~25/year) and avoid fee drag. The 1w EMA50 provides strong trend alignment, reducing whipsaws across bull/bear cycles. Volume spike confirms breakout momentum. Designed for BTC/ETH robustness with strict entry conditions (trend + volume + breakout) to minimize overtrading while capturing sustained moves. 12h timeframe targets 50-150 total trades over 4 years as per session guidance.
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
    
    # Calculate EMA50 on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get 1d data for Camarilla levels (primary HTF for pivot calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior day)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 20-bar average volume for confirmation on 12h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50(1w) and volume MA20
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 1.8x 20-bar average (balanced filter)
            volume_confirm = volume[i] > 1.8 * vol_ma20[i]
            
            # Long: price breaks above Camarilla R1 in uptrend with volume spike
            # Short: price breaks below Camarilla S1 in downtrend with volume spike
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema50_1w_aligned[i]) and volume_confirm
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema50_1w_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit when price moves back below 1w EMA50 (trend reversal)
            exit_signal = close[i] < ema50_1w_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit when price moves back above 1w EMA50 (trend reversal)
            exit_signal = close[i] > ema50_1w_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0