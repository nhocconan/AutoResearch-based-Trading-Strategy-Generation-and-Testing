#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeFilter_v1
Hypothesis: 6h Donchian(20) breakout filtered by weekly Camarilla pivot direction and volume confirmation.
In bullish weekly regime (price above weekly H3): long breakouts above 20-period high.
In bearish weekly regime (price below weekly L3): short breakouts below 20-period low.
Volume confirmation (1.5x average) filters false breakouts. Designed to work in both bull and bear markets.
Timeframe: 6h, uses 1d HTF for weekly pivot calculation via resampling (using actual 1d candles).
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === Weekly Camarilla pivot from 1d data (using last 5 trading days) ===
    # We approximate weekly using last 5x 1d bars (Mon-Fri)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate rolling 5-day high/low/close for weekly approximation
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Weekly Camarilla levels based on 5-day approximation
    range_5d = high_5d - low_5d
    h3_1d = close_5d + 1.1 * range_5d  # Weekly resistance 3
    l3_1d = close_5d - 1.1 * range_5d  # Weekly support 3
    
    # Align weekly levels to 6h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # === 6h Donchian channels (20-period) ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) 
            or np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume_now = volume[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        weekly_h3 = h3_1d_aligned[i]
        weekly_l3 = l3_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        # Determine weekly regime
        bullish_weekly = price > weekly_h3
        bearish_weekly = price < weekly_l3
        
        if position == 0:
            # Long: bullish weekly + price breaks above Donchian high + volume
            if bullish_weekly and price > dh and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: bearish weekly + price breaks below Donchian low + volume
            elif bearish_weekly and price < dl and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly turns bearish
            if price < dl or not bullish_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly turns bullish
            if price > dh or not bearish_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0