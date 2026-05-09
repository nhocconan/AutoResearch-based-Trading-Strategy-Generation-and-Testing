#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyChop_Trend_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian trend and chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly Donchian(20) for trend
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly Chop(14) for regime filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr14 * 14) / (highest_high - lowest_low)) / np.log10(14)
    
    # Daily volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily
    donchian_high_1d = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1w, donchian_low)
    chop_1d = align_htf_to_ltf(prices, df_1w, chop)
    vol_avg_1d = align_htf_to_ltf(prices, df_1w, vol_avg)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(chop_1d[i]) or np.isnan(vol_avg_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dh = donchian_high_1d[i]
        dl = donchian_low_1d[i]
        chop_val = chop_1d[i]
        vol_ok = volume[i] > vol_avg_1d[i] * 1.5
        
        if position == 0:
            # Trending market: chop < 38.2
            if chop_val < 38.2:
                # Long: break above weekly Donchian high with volume
                if close[i] > dh and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short: break below weekly Donchian low with volume
                elif close[i] < dl and vol_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: chop > 61.8 (range) or price retrace to midpoint
            midpoint = (dh + dl) / 2
            if chop_val > 61.8 or close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: chop > 61.8 (range) or price retrace to midpoint
            midpoint = (dh + dl) / 2
            if chop_val > 61.8 or close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals