#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RankVolume_Breakout_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Rank volume: volume / average volume (normalized)
    vol_ratio = volume / (pd.Series(volume).rolling(window=20, min_periods=20).mean().values + 1e-8)
    
    # Align all to 6h
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_1d_6h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_6h[i]) or np.isnan(vol_avg_1d_6h[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_1d_6h[i]
        vol_avg = vol_avg_1d_6h[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: high volume breakout above 1d EMA50
            if close[i] > trend and vol_ratio_val > 2.0:
                signals[i] = 0.25
                position = 1
            # Short: high volume breakdown below 1d EMA50
            elif close[i] < trend and vol_ratio_val > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below trend or volume drops
            if close[i] < trend or vol_ratio_val < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above trend or volume drops
            if close[i] > trend or vol_ratio_val < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals