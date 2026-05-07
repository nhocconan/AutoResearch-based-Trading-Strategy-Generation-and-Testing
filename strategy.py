#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Daily high/low for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla R3 and S3 from previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    camarilla_r3_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 2
    camarilla_s3_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Align daily Camarilla levels to 1h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # 1h volume spike filter: > 2.5x 24-period average
    vol_ma_1h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike_1h = volume > 2.5 * vol_ma_1h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 20)  # Wait for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with volume spike and price above 4h EMA20 (uptrend)
            if (close[i] > camarilla_r3_1d_aligned[i] and 
                vol_spike_1h[i] and 
                close[i] > ema20_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Break below S3 with volume spike and price below 4h EMA20 (downtrend)
            elif (close[i] < camarilla_s3_1d_aligned[i] and 
                  vol_spike_1h[i] and 
                  close[i] < ema20_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Price below S3 or below 4h EMA20 (trend reversal)
            if close[i] < camarilla_s3_1d_aligned[i] or close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price above R3 or above 4h EMA20 (trend reversal)
            if close[i] > camarilla_r3_1d_aligned[i] or close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals