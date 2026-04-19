#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot reversal and daily volume confirmation.
# Uses weekly pivot points (R1/S1) to identify key levels: long when price retraces to S1 with bullish daily candle,
# short when price retraces to R1 with bearish daily candle. Weekly pivot provides structure from higher timeframe,
# daily volume confirms participation. Works in both bull and bear markets by fading extremes at key levels.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "6h_1w_Pivot_R1S1_DailyVolume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Get daily data for volume and candle direction (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate daily volume filter: volume > 1.5 * 20-day average
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (volume_ma_1d * 1.5)
    
    # Calculate daily candle direction: 1 for bullish (close > open), -1 for bearish
    daily_bullish = close_1d > open_1d
    daily_bearish = close_1d < open_1d
    
    # Align daily filters to 6h timeframe
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d.astype(float))
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(volume_filter_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Current price levels
        curr_price = close[i]
        r1_level = r1_1w_aligned[i]
        s1_level = s1_1w_aligned[i]
        
        # Check if price is near weekly S1 (within 0.5%) for long
        near_s1 = abs(curr_price - s1_level) / s1_level < 0.005
        # Check if price is near weekly R1 (within 0.5%) for short
        near_r1 = abs(curr_price - r1_level) / r1_level < 0.005
        
        if position == 0:
            # Long when price retraces to S1 with bullish daily candle and volume
            if near_s1 and daily_bullish_aligned[i] > 0.5 and volume_filter_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short when price retraces to R1 with bearish daily candle and volume
            elif near_r1 and daily_bearish_aligned[i] > 0.5 and volume_filter_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price moves above weekly pivot or stops bullish
            if curr_price > pivot_1w_aligned[i] or daily_bullish_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price moves below weekly pivot or stops bearish
            if curr_price < pivot_1w_aligned[i] or daily_bearish_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals