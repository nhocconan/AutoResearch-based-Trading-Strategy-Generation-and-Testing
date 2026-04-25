#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_v2
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA50 trend filter and volume spike confirmation.
Targets 15-25 trades/year per symbol to minimize fee drag while capturing sustained moves in both bull and bear markets.
Uses strict volume confirmation (>2.5x ATR-scaled volume mean) and trend alignment (price >1.0% above/below weekly EMA50).
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
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate EMA50 on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate ATR(14) on 1w for dynamic volume threshold
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), np.abs(low_1w[1:] - close_1w[:-1])))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    # Get 1d data for Camarilla levels (using previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d)  # R1 = C + 1.1*(H-L)
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d)  # S1 = C - 1.1*(H-L)
    
    # Align Camarilla levels to 1d timeframe (use previous bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate dynamic volume threshold: 2.5x ATR-scaled volume mean (stricter confirmation)
    vol_atr_ratio = volume / (atr14_1w_aligned * close + 1e-10)  # Avoid division by zero
    vol_threshold = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).mean().values * 2.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50, ATR, volume threshold
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_threshold[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Dynamic volume confirmation: current volume > threshold * average volume
            vol_avg = pd.Series(volume).rolling(window=20, min_periods=1).mean().iloc[i]
            vol_confirm = volume[i] > vol_threshold[i] * vol_avg
            
            # Long: price breaks above Camarilla R1 in uptrend (price > 1w EMA50 + 1.0%) with volume confirmation
            # Short: price breaks below Camarilla S1 in downtrend (price < 1w EMA50 - 1.0%) with volume confirmation
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema50_1w_aligned[i] * 1.010) and vol_confirm
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema50_1w_aligned[i] * 0.990) and vol_confirm
            
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
            # Exit when price moves back below 1w EMA50 (trend reversal)
            exit_signal = close[i] < ema50_1w_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1w EMA50 (trend reversal)
            exit_signal = close[i] > ema50_1w_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_v2"
timeframe = "1d"
leverage = 1.0