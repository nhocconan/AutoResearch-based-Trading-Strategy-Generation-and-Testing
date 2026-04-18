#!/usr/bin/env python3
"""
1d Donchian Breakout with Volume Confirmation and Weekly Trend Filter
Hypothesis: Price breaking above/below Donchian Channel (20-day) with volume confirmation 
(volume > 1.5x 20-day average) and weekly trend alignment (price above/below weekly EMA34) 
indicates strong momentum. Weekly EMA34 filter reduces false breakouts in counter-trend moves.
Target: 10-25 trades/year to minimize fee drain on daily timeframe.
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
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Warmup for indicators (max of 20,20,34)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(weekly_ema34_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        weekly_trend = weekly_ema34_aligned[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume + weekly uptrend
            if price > upper and vol_conf and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume + weekly downtrend
            elif price < lower and vol_conf and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to weekly EMA34 or breaks below Donchian lower
            if price < weekly_trend or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to weekly EMA34 or breaks above Donchian upper
            if price > weekly_trend or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_Volume_WeeklyEMA34"
timeframe = "1d"
leverage = 1.0