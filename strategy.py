#!/usr/bin/env python3
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
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from daily data
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Weekly pivot points: P = (H + L + C)/3
    weekly_pivot = (high_5d + low_5d + close_5d) / 3.0
    # Weekly resistance and support levels
    weekly_r1 = 2 * weekly_pivot - low_5d
    weekly_s1 = 2 * weekly_pivot - high_5d
    
    # Align weekly pivot levels to daily timeframe
    weekly_pivot_1d = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_1d = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_1d = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Get weekly data for trend filter (EMA21)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # Need weekly pivot, weekly EMA21, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_1d[i]) or 
            np.isnan(weekly_r1_1d[i]) or 
            np.isnan(weekly_s1_1d[i]) or 
            np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA21
        price_above_ema = close[i] > ema21_1w_aligned[i]
        price_below_ema = close[i] < ema21_1w_aligned[i]
        
        # Price relative to weekly pivot levels
        price_above_r1 = close[i] > weekly_r1_1d[i]
        price_below_s1 = close[i] < weekly_s1_1d[i]
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume and above weekly EMA21
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume and below weekly EMA21
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly pivot OR below weekly EMA21
            if (close[i] < weekly_pivot_1d[i]) or (close[i] < ema21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly pivot OR above weekly EMA21
            if (close[i] > weekly_pivot_1d[i]) or (close[i] > ema21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_Breakout_EMA21_Volume"
timeframe = "1d"
leverage = 1.0