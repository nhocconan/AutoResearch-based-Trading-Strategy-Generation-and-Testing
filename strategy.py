#!/usr/bin/env python3
"""
12h_Alligator_Jaw_Crossover_1dTrend_1wFilter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips crossover) on 12h with daily trend filter and weekly EMA filter.
Trades in direction of higher timeframe trends to avoid whipsaws. Alligator crossover provides timely entry
with built-in smoothing. Targets 20-30 trades/year to minimize fee drag while capturing trend changes.
Works in bull (trend following) and bear (avoids counter-trend trades via filters).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get weekly data for additional filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly EMA(50) for filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 12h: Jaw(13,8), Teeth(8,5), Lips(5,3)
    # Jaw: 13-period SMMA smoothed 8 periods
    # Teeth: 8-period SMMA smoothed 5 periods  
    # Lips: 5-period SMMA smoothed 3 periods
    # Using EMA as proxy for SMMA for simplicity and responsiveness
    
    # Jaw (blue line): 13-period EMA, then smoothed 8 more
    jaw_raw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth (red line): 8-period EMA, then smoothed 5 more
    teeth_raw = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips (green line): 5-period EMA, then smoothed 3 more
    lips_raw = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips_raw).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Daily trend filter: price above/below EMA(34)
        daily_bullish = close[i] > ema_34_1d_aligned[i]
        daily_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Weekly filter: price above/below EMA(50)
        weekly_bullish = close[i] > ema_50_1w_aligned[i]
        weekly_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish alignment
        # Lips < Teeth < Jaw = bearish alignment
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Alligator crossover signals
        lips_cross_teeth_up = (lips[i-1] <= teeth[i-1]) and (lips[i] > teeth[i])
        lips_cross_teeth_down = (lips[i-1] >= teeth[i-1]) and (lips[i] < teeth[i])
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: Alligator crossover in alignment with daily/weekly trends
        long_entry = vol_confirm and bullish_alignment and daily_bullish and weekly_bullish and lips_cross_teeth_up
        short_entry = vol_confirm and bearish_alignment and daily_bearish and weekly_bearish and lips_cross_teeth_down
        
        # Exit logic: opposing crossover or trend failure
        long_exit = lips_cross_teeth_down or (not daily_bullish) or (not weekly_bullish)
        short_exit = lips_cross_teeth_up or (not daily_bearish) or (not weekly_bearish)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Alligator_Jaw_Crossover_1dTrend_1wFilter"
timeframe = "12h"
leverage = 1.0