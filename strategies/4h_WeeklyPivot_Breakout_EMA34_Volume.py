#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation (using 5 daily bars)
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
    
    # Align weekly pivot levels to 4h timeframe
    weekly_pivot_4h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_4h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_4h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Get 4h data for trend filter (EMA34)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume filter: current volume > 1.5 * 30-period average
    volume_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need weekly pivot, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_4h[i]) or 
            np.isnan(weekly_r1_4h[i]) or 
            np.isnan(weekly_s1_4h[i]) or 
            np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(volume_ma30[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma30[i])
        
        # Trend filter: price above/below 4h EMA34
        price_above_ema = close[i] > ema34_4h_aligned[i]
        price_below_ema = close[i] < ema34_4h_aligned[i]
        
        # Price relative to weekly pivot levels
        price_above_r1 = close[i] > weekly_r1_4h[i]
        price_below_s1 = close[i] < weekly_s1_4h[i]
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume and above 4h EMA34
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume and below 4h EMA34
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly pivot OR below 4h EMA34
            if (close[i] < weekly_pivot_4h[i]) or (close[i] < ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly pivot OR above 4h EMA34
            if (close[i] > weekly_pivot_4h[i]) or (close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WeeklyPivot_Breakout_EMA34_Volume"
timeframe = "4h"
leverage = 1.0