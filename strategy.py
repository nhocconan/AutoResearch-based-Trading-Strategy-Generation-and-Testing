#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and Daily Trend Filter
Hypothesis: Donchian(20) breakouts with volume > 1.5x average and daily EMA34 trend filter 
capture strong momentum moves while reducing false breakouts. Works in both bull and bear 
markets by following the daily trend direction. Target: 20-40 trades/year to minimize fee drag.
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
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA20 for exit
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    # Load daily EMA34 for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Warmup for indicators (max of 20,20,34)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        ema34 = ema34_1d_aligned[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + above daily EMA34
            if price > upper and vol_conf and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume + below daily EMA34
            elif price < lower and vol_conf and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to EMA20 or breaks below Donchian low
            if price < ema20[i] or price < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to EMA20 or breaks above Donchian high
            if price > ema20[i] or price > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_DailyEMA34"
timeframe = "4h"
leverage = 1.0