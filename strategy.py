#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Long when price breaks above 20-period high AND weekly pivot trend is up AND volume > 1.5x average
# Short when price breaks below 20-period low AND weekly pivot trend is down AND volume > 1.5x average
# Exit when price crosses back through the 20-period midpoint or opposite breakout occurs
# Weekly pivot trend: price > weekly pivot = bullish, price < weekly pivot = bearish
# This captures strong momentum moves in alignment with higher timeframe structure
# Volume ensures conviction, reducing false breakouts
# Target: 15-30 trades/year by requiring confluence of breakout, trend, and volume

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot point (standard formula)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    weekly_pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # Calculate Donchian channels (20-period) on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 20-period highest high and lowest low
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (highest_20 + lowest_20) / 2.0
    
    # Calculate 6h volume moving average (20-period)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = close[i]
        highest = highest_20[i]
        lowest = lowest_20[i]
        midpoint = midpoint_20[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_ma_val = vol_ma[i]
        current_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = current_volume > 1.5 * vol_ma_val
        
        if position == 0:
            # Long breakout: price > 20-period high AND price > weekly pivot (bullish trend) AND volume confirmation
            if price > highest and price > weekly_pivot_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < 20-period low AND price < weekly pivot (bearish trend) AND volume confirmation
            elif price < lowest and price < weekly_pivot_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below 20-period midpoint OR breaks below 20-period low
                if price < midpoint or price < lowest:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above 20-period midpoint OR breaks above 20-period high
                if price > midpoint or price > highest:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0