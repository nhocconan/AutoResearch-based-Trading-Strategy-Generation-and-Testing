#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot levels as support/resistance with volume confirmation
# - Long when price rebounds from weekly S1/S2 with volume > 1.3x 50-period average and close > open
# - Short when price reverses from weekly R1/R2 with volume > 1.3x 50-period average and close < open
# - Uses weekly pivot levels as institutional reference points that work in both bull and bear markets
# - Volume confirmation ensures institutional participation at key levels
# - Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and cost
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Load weekly data once before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points
    pivot = (high_w + low_w + close_w) / 3
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    
    # Volume filter: 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if np.isnan(vol_ma[i]):
            continue
        
        # Get previous week's pivot levels for current week
        idx_w = i // (7 * 24 * 60 // 6)  # Approximate weekly index from 6h bars
        if idx_w < 1:
            continue
            
        # Previous week's levels
        piv_prev = pivot[idx_w-1]
        r1_prev = r1[idx_w-1]
        s1_prev = s1[idx_w-1]
        r2_prev = r2[idx_w-1]
        s2_prev = s2[idx_w-1]
        
        # Create arrays for alignment (constant values for the week)
        pivot_arr = np.full(len(df_w), piv_prev)
        r1_arr = np.full(len(df_w), r1_prev)
        s1_arr = np.full(len(df_w), s1_prev)
        r2_arr = np.full(len(df_w), r2_prev)
        s2_arr = np.full(len(df_w), s2_prev)
        
        # Align to 6h timeframe
        pivot_6h = align_htf_to_ltf(prices, df_w, pivot_arr)[i]
        r1_6h = align_htf_to_ltf(prices, df_w, r1_arr)[i]
        s1_6h = align_htf_to_ltf(prices, df_w, s1_arr)[i]
        r2_6h = align_htf_to_ltf(prices, df_w, r2_arr)[i]
        s2_6h = align_htf_to_ltf(prices, df_w, s2_arr)[i]
        
        if position == 0:
            # Long: Price near support with bullish candle and volume
            if ((low[i] <= s1_6h * 1.005 or low[i] <= s2_6h * 1.005) and  # Near S1/S2
                close[i] > open_price[i] and  # Bullish candle
                volume[i] > vol_ma[i] * 1.3):  # Volume confirmation
                position = 1
                signals[i] = position_size
            # Short: Price near resistance with bearish candle and volume
            elif ((high[i] >= r1_6h * 0.995 or high[i] >= r2_6h * 0.995) and  # Near R1/R2
                  close[i] < open_price[i] and  # Bearish candle
                  volume[i] > vol_ma[i] * 1.3):  # Volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price reaches pivot or opposite resistance
            if close[i] >= pivot_6h or close[i] >= r1_6h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price reaches pivot or opposite support
            if close[i] <= pivot_6h or close[i] <= s1_6h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_1w_PivotReversal_Volume"
timeframe = "6h"
leverage = 1.0