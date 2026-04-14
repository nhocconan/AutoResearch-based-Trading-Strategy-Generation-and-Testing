# 6h strategy using monthly pivot levels for institutional support/resistance with volume confirmation
# Monthly pivots provide strong institutional reference points that work across market regimes
# Volume ensures institutional participation at key levels
# Long when bouncing from monthly S1/S2 with bullish candle and volume > 1.3x average
# Short when rejected at monthly R1/R2 with bearish candle and volume > 1.3x average
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Load monthly data once before loop
    df_m = get_htf_data(prices, '1M')
    if len(df_m) < 2:
        return np.zeros(n)
    
    high_m = df_m['high'].values
    low_m = df_m['low'].values
    close_m = df_m['close'].values
    
    # Calculate monthly pivot points
    pivot = (high_m + low_m + close_m) / 3
    r1 = 2 * pivot - low_m
    s1 = 2 * pivot - high_m
    r2 = pivot + (high_m - low_m)
    s2 = pivot - (high_m - low_m)
    
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
        
        # Get previous month's pivot levels for current month
        idx_m = i // (30 * 24 * 60 // 6)  # Approximate monthly index from 6h bars
        if idx_m < 1:
            continue
            
        # Previous month's levels
        piv_prev = pivot[idx_m-1]
        r1_prev = r1[idx_m-1]
        s1_prev = s1[idx_m-1]
        r2_prev = r2[idx_m-1]
        s2_prev = s2[idx_m-1]
        
        # Create arrays for alignment (constant values for the month)
        pivot_arr = np.full(len(df_m), piv_prev)
        r1_arr = np.full(len(df_m), r1_prev)
        s1_arr = np.full(len(df_m), s1_prev)
        r2_arr = np.full(len(df_m), r2_prev)
        s2_arr = np.full(len(df_m), s2_prev)
        
        # Align to 6h timeframe
        pivot_6h = align_htf_to_ltf(prices, df_m, pivot_arr)[i]
        r1_6h = align_htf_to_ltf(prices, df_m, r1_arr)[i]
        s1_6h = align_htf_to_ltf(prices, df_m, s1_arr)[i]
        r2_6h = align_htf_to_ltf(prices, df_m, r2_arr)[i]
        s2_6h = align_htf_to_ltf(prices, df_m, s2_arr)[i]
        
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

name = "6h_1M_PivotReversal_Volume"
timeframe = "6h"
leverage = 1.0