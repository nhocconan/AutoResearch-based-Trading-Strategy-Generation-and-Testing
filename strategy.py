#!/usr/bin/env python3
name = "1d_WeeklyDonchian_Breakout_Trend_Force_v1"
timeframe = "1d"
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
    
    # Weekly Donchian (20) - upper and lower bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels
    donch_high = np.full(len(df_1w), np.nan)
    donch_low = np.full(len(df_1w), np.nan)
    for i in range(19, len(df_1w)):
        donch_high[i] = np.max(high_1w[i-19:i+1])
        donch_low[i] = np.min(low_1w[i-19:i+1])
    
    # Align to daily timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Daily trend filter: EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: break above weekly Donchian high + weekly uptrend + volume
            if close[i] > donch_high_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly Donchian low + weekly downtrend + volume
            elif close[i] < donch_low_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price below weekly Donchian low or trend breaks
            if close[i] < donch_low_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price above weekly Donchian high or trend breaks
            if close[i] > donch_high_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals