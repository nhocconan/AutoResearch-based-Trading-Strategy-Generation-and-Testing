#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot points (using prior week's data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using prior week's OHLC
    # We'll use the prior week's data by shifting the daily data
    week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(5).values
    week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(5).values
    week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(5).values
    
    # Weekly pivot points (classic)
    weekly_pivot = (week_high + week_low + week_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - week_low
    weekly_s1 = 2 * weekly_pivot - week_high
    weekly_r2 = weekly_pivot + (week_high - week_low)
    weekly_s2 = weekly_pivot - (week_high - week_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_6h = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_6h = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need weekly data and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_6h[i]) or 
            np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or 
            np.isnan(weekly_r2_6h[i]) or 
            np.isnan(weekly_s2_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Price relative to weekly pivot levels
        price_above_r2 = close[i] > weekly_r2_6h[i]
        price_below_s2 = close[i] < weekly_s2_6h[i]
        price_between_r1_r2 = (close[i] > weekly_r1_6h[i]) & (close[i] < weekly_r2_6h[i])
        price_between_s1_s2 = (close[i] > weekly_s1_6h[i]) & (close[i] < weekly_s2_6h[i])
        
        if position == 0:
            # Long: Break above R2 with volume (strong bullish breakout)
            if price_above_r2 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below S2 with volume (strong bearish breakdown)
            elif price_below_s2 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below R1 (loss of bullish momentum)
            if close[i] < weekly_r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above S1 (loss of bearish momentum)
            if close[i] > weekly_s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R2S2_Breakout_Volume"
timeframe = "6h"
leverage = 1.0