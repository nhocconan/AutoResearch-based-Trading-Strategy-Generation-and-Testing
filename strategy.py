#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Use 1d Camarilla R3/S3 levels as price structure, with 1d EMA(34) for trend filter and volume spike for confirmation on 12h timeframe.
Long when price breaks above 1d Camarilla R3 and close > 1d EMA(34) and volume > 2x 20-period average.
Short when price breaks below 1d Camarilla S3 and close < 1d EMA(34) and volume > 2x 20-period average.
This targets meaningful breakouts with trend alignment and avoids overtrading via strict volume confirmation.
"""
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "12h"
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
    
    # Get 1d data for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # R3 = High + 1.1 * (High - Low) / 2
    # S3 = Low - 1.1 * (High - Low) / 2
    camarilla_r3 = df_1d['high'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_s3 = df_1d['low'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_r3_vals = camarilla_r3.values
    camarilla_s3_vals = camarilla_s3.values
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_vals)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_vals)
    
    # Calculate 1d EMA34 for trend
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 2 bars between trades (24 hours on 12h TF) to reduce frequency
            if bars_since_exit < 2:
                continue
                
            # Long: price breaks above 1d Camarilla R3 + close > 1d EMA + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below 1d Camarilla S3 + close < 1d EMA + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Camarilla level or trend reversal
            if position == 1 and (close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and (close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals