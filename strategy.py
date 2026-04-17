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
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly data
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (classic)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_r1 = 2 * weekly_pivot - low_1w
    weekly_s1 = 2 * weekly_pivot - high_1w
    
    # Calculate daily EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly pivot levels to 12h timeframe
    weekly_pivot_12h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_12h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_12h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Align daily EMA200 to 12h timeframe
    ema200_12h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: current volume > 2.0 * 50-period average
    volume_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need daily EMA200, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_12h[i]) or 
            np.isnan(weekly_r1_12h[i]) or 
            np.isnan(weekly_s1_12h[i]) or 
            np.isnan(ema200_12h[i]) or 
            np.isnan(volume_ma50[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (2.0 * volume_ma50[i])
        
        # Trend filter: price above/below daily EMA200
        price_above_ema = close[i] > ema200_12h[i]
        price_below_ema = close[i] < ema200_12h[i]
        
        # Price relative to weekly pivot levels
        price_above_r1 = close[i] > weekly_r1_12h[i]
        price_below_s1 = close[i] < weekly_s1_12h[i]
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume and above daily EMA200
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume and below daily EMA200
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly pivot OR below daily EMA200
            if (close[i] < weekly_pivot_12h[i]) or (close[i] < ema200_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly pivot OR above daily EMA200
            if (close[i] > weekly_pivot_12h[i]) or (close[i] > ema200_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_Breakout_EMA200_Volume"
timeframe = "12h"
leverage = 1.0