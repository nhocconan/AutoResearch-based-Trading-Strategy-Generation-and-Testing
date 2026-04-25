#!/usr/bin/env python3
"""
1d_Camarilla_H4_Trend_VolumeSpike_v1
Hypothesis: Daily Camarilla R3/S3 breakouts with 1-week EMA50 trend filter and volume spike confirmation.
Primary timeframe 1d targets 7-25 trades/year (30-100 total over 4 years) to minimize fee drag.
Weekly EMA50 provides strong trend alignment that works in both bull and bear markets by filtering counter-trend breakouts.
Volume spike (>2.0x 20-bar average) confirms breakout momentum and reduces false signals.
Designed for BTC/ETH with discrete sizing (0.30) to manage drawdown and avoid overtrading.
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
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    # Need 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla R3 and S3 levels from previous 1d bar
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 1d timeframe (use previous bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 20-bar average volume for confirmation on 1d
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50, volume MA20
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 2.0x 20-bar average
            volume_confirm = volume[i] > 2.0 * vol_ma20[i]
            
            # Long: price breaks above Camarilla R3 in uptrend (price > 1w EMA50) with volume spike
            # Short: price breaks below Camarilla S3 in downtrend (price < 1w EMA50) with volume spike
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema50_1w_aligned[i]) and volume_confirm
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema50_1w_aligned[i]) and volume_confirm
            
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

name = "1d_Camarilla_H4_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0