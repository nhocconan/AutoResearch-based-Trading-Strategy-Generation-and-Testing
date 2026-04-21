#!/usr/bin/env python3
"""
4h_1d_PriceChannel_Breakout_Volume_Tight_v2
Hypothesis: Use 1d Donchian(20) breakout with volume confirmation and 4h EMA21 trend filter.
Long when price breaks above 1d Donchian upper band + volume > 2x 20-period avg + 4h EMA21 rising.
Short when price breaks below 1d Donchian lower band + volume > 2x 20-period avg + 4h EMA21 falling.
Exit when price crosses 1d EMA21. Designed for <30 trades/year to minimize fee drag.
Works in bull (follows 1d breakouts) and bear (avoids false breaks via volume+trend filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Donchian channels and EMA21
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Donchian(20) channels
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA21 for exit
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align to 4h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Load 4h data for EMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema21_1d_aligned[i]) or np.isnan(ema21_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # 4h EMA21 trend filter: rising for long, falling for short
        if i >= 1:
            ema21_4h_prev = ema21_4h_aligned[i-1]
            ema21_4h_curr = ema21_4h_aligned[i]
            ema_rising = ema21_4h_curr > ema21_4h_prev
            ema_falling = ema21_4h_curr < ema21_4h_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long conditions: break above Donchian upper + volume + rising EMA
            if price > highest_20_aligned[i] and volume_ok and ema_rising:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian lower + volume + falling EMA
            elif price < lowest_20_aligned[i] and volume_ok and ema_falling:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 1d EMA21
            if price < ema21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 1d EMA21
            if price > ema21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_PriceChannel_Breakout_Volume_Tight_v2"
timeframe = "4h"
leverage = 1.0