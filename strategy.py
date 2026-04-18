#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Confirmation and 1-week Trend Filter
Hypothesis: Price breaking above/below Donchian Channel (20-period high/low) on 12h timeframe,
with volume confirmation (volume > 1.8x average) and 1-week trend direction (price above/below 200 EMA),
indicates strong momentum with filtered false breakouts. Uses weekly trend to avoid counter-trend trades.
Target: 15-30 trades/year to minimize fee drag and work in both bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period high/low)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    # 1-week trend: 200 EMA on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators (max of 20,20,200)
    
    for i in range(start_idx, n):
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_donch = high_roll[i]
        lower_donch = low_roll[i]
        vol_conf = vol_ratio[i] > 1.8
        weekly_trend_up = price > ema200_1w_aligned[i]
        weekly_trend_down = price < ema200_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume + weekly uptrend
            if price > upper_donch and vol_conf and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume + weekly downtrend
            elif price < lower_donch and vol_conf and weekly_trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to midpoint of Donchian channel
            midpoint = (upper_donch + lower_donch) * 0.5
            if price < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to midpoint of Donchian channel
            midpoint = (upper_donch + lower_donch) * 0.5
            if price > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0