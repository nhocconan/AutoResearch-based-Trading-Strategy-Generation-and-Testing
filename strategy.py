#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # Weekly data for trend and Camarilla
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (R3, S3) from previous day
    # R3 = Close + 1.1 * (High - Low) / 2
    # S3 = Close - 1.1 * (High - Low) / 2
    camarilla_r3 = close_1d + (1.1 * (high_1d - low_1d) / 2)
    camarilla_s3 = close_1d - (1.1 * (high_1d - low_1d) / 2)
    
    # Align all to 12h
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike on 12h: current volume > 2.0x 50-period average
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + weekly uptrend + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 + weekly downtrend + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 or weekly trend turns down
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above R3 or weekly trend turns up
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals