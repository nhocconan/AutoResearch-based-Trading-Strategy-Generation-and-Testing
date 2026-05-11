#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
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
    
    # Daily close for Camarilla and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate previous day's Camarilla levels (shifted by 1)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla R1, S1, R3, S3
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Align Camarilla levels to 4h
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection on 4h
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # ensure EMA34 and Camarilla ready
    
    for i in range(start_idx, n):
        if np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or np.isnan(ema_34_4h[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close crosses above R1 with volume spike and price > EMA34
            if close[i] > R1_4h[i] and close[i-1] <= R1_4h[i-1] and volume_spike[i] and close[i] > ema_34_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close crosses below S1 with volume spike and price < EMA34
            elif close[i] < S1_4h[i] and close[i-1] >= S1_4h[i-1] and volume_spike[i] and close[i] < ema_34_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close crosses below S1
            if close[i] < S1_4h[i] and close[i-1] >= S1_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close crosses above R1
            if close[i] > R1_4h[i] and close[i-1] <= R1_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals