#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Spike"
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
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h EMA20 for trend filter
    ema20_12h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d Camarilla levels: R3, S3 from previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    camarilla_r3_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 2
    camarilla_s3_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # 12h volume spike: > 2.0x 30-period average
    vol_ma_12h = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike_12h = volume > 2.0 * vol_ma_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 30)  # Wait for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with volume spike and price above EMA20
            if (close[i] > camarilla_r3_1d_aligned[i] and vol_spike_12h[i] and 
                close[i] > ema20_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume spike and price below EMA20
            elif (close[i] < camarilla_s3_1d_aligned[i] and vol_spike_12h[i] and 
                  close[i] < ema20_12h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below S3
            if close[i] < camarilla_s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above R3
            if close[i] > camarilla_r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakouts with volume confirmation and EMA20 trend filter.
# Works in bull/bear markets by capturing breakouts in trending conditions.
# Target: 20-40 trades/year to minimize fee drag. Position size 0.25 limits risk.