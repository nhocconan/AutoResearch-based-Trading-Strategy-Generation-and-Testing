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
    
    # Get daily data for Williams %R (overbought/oversold)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_1d) / (highest_high - lowest_low)) * -100
    
    # Use previous day's Williams %R (avoid look-ahead)
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    
    # Align daily Williams %R to 6h timeframe
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r_prev)
    
    # Get weekly data for trend filter (EWMA crossover)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(8) and EMA(21) for trend
    ema8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMAs to 6h timeframe
    ema8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema8_1w)
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need Williams %R, EMAs and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_6h[i]) or 
            np.isnan(ema8_1w_aligned[i]) or 
            np.isnan(ema21_1w_aligned[i]) or
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Weekly trend filter: EMA8 > EMA21 for uptrend, EMA8 < EMA21 for downtrend
        weekly_uptrend = ema8_1w_aligned[i] > ema21_1w_aligned[i]
        weekly_downtrend = ema8_1w_aligned[i] < ema21_1w_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume AND weekly uptrend
            if williams_r_6h[i] < -80 and volume_filter and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume AND weekly downtrend
            elif williams_r_6h[i] > -20 and volume_filter and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -50 or trend changes
            if williams_r_6h[i] > -50 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -50 or trend changes
            if williams_r_6h[i] < -50 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_EMACrossover_VolumeFilter"
timeframe = "6h"
leverage = 1.0