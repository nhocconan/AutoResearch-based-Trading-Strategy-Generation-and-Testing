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
    
    # Weekly Donchian (20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donch_high = np.full(len(df_1w), np.nan)
    donch_low = np.full(len(df_1w), np.nan)
    for i in range(19, len(df_1w)):
        donch_high[i] = np.max(high_1w[i-19:i+1])
        donch_low[i] = np.min(low_1w[i-19:i+1])
    
    dh = align_htf_to_ltf(prices, df_1w, donch_high)
    dl = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Weekly EMA(34) trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: current volume > 1.5 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(dh[i]) or np.isnan(dl[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above weekly Donchian high with weekly uptrend and volume filter
            if close[i] > dh[i] and close[i] > ema34_1w_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly Donchian low with weekly downtrend and volume filter
            elif close[i] < dl[i] and close[i] < ema34_1w_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly Donchian low
            if close[i] < dl[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly Donchian high
            if close[i] > dh[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals