#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    R4 = prev_close + (prev_high - prev_low) * 1.5
    R3 = prev_close + (prev_high - prev_low) * 1.25
    R2 = prev_close + (prev_high - prev_low) * 1.166
    R1 = prev_close + (prev_high - prev_low) * 1.083
    S1 = prev_close - (prev_high - prev_low) * 1.083
    S2 = prev_close - (prev_high - prev_low) * 1.166
    S3 = prev_close - (prev_high - prev_low) * 1.25
    S4 = prev_close - (prev_high - prev_low) * 1.5
    
    # Align to 12h
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with uptrend and volume
            if close[i] > R3_12h[i] and close[i] > ema34_12h[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with downtrend and volume
            elif close[i] < S3_12h[i] and close[i] < ema34_12h[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Break below S3 or trend reversal
            if close[i] < S3_12h[i] or close[i] < ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Break above R3 or trend reversal
            if close[i] > R3_12h[i] or close[i] > ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals