#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and 1d Trend Filter
Hypothesis: Price breaking above/below Donchian Channel (20-period high/low) with volume confirmation 
(volume > 1.5x average) and 1d EMA34 trend filter indicates strong momentum. 
Donchian breakouts capture strong moves, volume confirms institutional interest, 
and 1d EMA34 filters for primary trend alignment. Works in both bull and bear markets 
by following the higher timeframe trend.
Target: 20-40 trades/year to minimize fee drain.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period high/low)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    # 1d EMA34 for trend filter (loaded once, aligned)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Warmup for indicators (max of 20,20,34)
    
    for i in range(start_idx, n):
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_roll[i]
        lower = low_roll[i]
        vol_conf = vol_ratio[i] > 1.5
        trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume + uptrend
            if price > upper and vol_conf and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume + downtrend
            elif price < lower and vol_conf and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to midpoint or trend reverses
            midpoint = (upper + lower) * 0.5
            if price < midpoint or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to midpoint or trend reverses
            midpoint = (upper + lower) * 0.5
            if price > midpoint or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0