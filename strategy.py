#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_Volume_Confirmation_v1
Hypothesis: On 6h timeframe, buy breakouts above 20-period Donchian high when weekly pivot trend is up and volume confirms; sell breakdowns below 20-period Donchian low when weekly pivot trend is down and volume confirms. Weekly pivot trend is determined by price position relative to weekly pivot point (PP). This strategy captures momentum in both bull and bear markets by aligning with higher timeframe trend while using volume to filter false breakouts. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for weekly pivot calculation (using daily data to approximate weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points using last 5 days (approximation)
    high_5d = df_1d['high'].rolling(window=5, min_periods=5).max()
    low_5d = df_1d['low'].rolling(window=5, min_periods=5).min()
    close_5d = df_1d['close'].rolling(window=5, min_periods=5).last()
    
    # Weekly pivot point (PP) = (H + L + C)/3
    pp = (high_5d + low_5d + close_5d) / 3.0
    # Align to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    
    # Weekly trend: up if close > PP, down if close < PP
    weekly_trend_up = close_5d > pp
    weekly_trend_down = close_5d < pp
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1d, weekly_trend_up.values)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1d, weekly_trend_down.values)
    
    # Donchian channels on 6h data (20-period)
    high_20 = prices['high'].rolling(window=20, min_periods=20).max()
    low_20 = prices['low'].rolling(window=20, min_periods=20).min()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(weekly_trend_up_aligned[i]) or 
            np.isnan(weekly_trend_down_aligned[i]) or np.isnan(high_20.iloc[i]) or 
            np.isnan(low_20.iloc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long: breakout above Donchian high with up weekly trend and volume
            if (price_high > high_20.iloc[i] and 
                weekly_trend_up_aligned[i] and volume_ok):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low with down weekly trend and volume
            elif (price_low < low_20.iloc[i] and 
                  weekly_trend_down_aligned[i] and volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below Donchian low or weekly trend turns down
            if price_close < low_20.iloc[i] or not weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above Donchian high or weekly trend turns up
            if price_close > high_20.iloc[i] or not weekly_trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_Volume_Confirmation_v1"
timeframe = "6h"
leverage = 1.0