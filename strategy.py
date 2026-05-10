#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Reversal_With_Volume
# Hypothesis: Trade reversals at Camarilla pivot levels from 1d timeframe during low volatility regimes.
# Uses Camarilla H4/L4 levels (strong support/resistance) with volume confirmation and chop filter.
# Works in bull markets by buying dips to support, in bear markets by selling rallies to resistance.
# Targets 15-25 trades/year to minimize fee drag on 12h timeframe.

name = "12h_Camarilla_Pivot_Reversal_With_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # 1d data for Camarilla pivots and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # Using previous day's values (no look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_h4 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l4 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Chopiness index for regime filter (using 1d data)
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        atr = np.zeros(len(close_arr))
        tr = np.zeros(len(close_arr))
        for i in range(1, len(close_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]),
                       abs(low_arr[i] - close_arr[i-1]))
        tr[0] = high_arr[0] - low_arr[0]
        
        # Calculate ATR using Wilder's smoothing
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (window-1) + tr[i]) / window
        
        # Calculate Chop
        sum_tr = np.zeros(len(close_arr))
        max_h = np.zeros(len(close_arr))
        min_l = np.zeros(len(close_arr))
        
        for i in range(window, len(close_arr)):
            sum_tr[i] = np.sum(tr[i-window+1:i+1])
            max_h[i] = np.max(high_arr[i-window+1:i+1])
            min_l[i] = np.min(low_arr[i-window+1:i+1])
            if max_h[i] - min_l[i] > 0:
                chop[i] = 100 * np.log10(sum_tr[i] / (max_h[i] - min_l[i])) / np.log10(window)
            else:
                chop[i] = 50
        
        # Fill beginning values
        for i in range(window):
            chop[i] = 50
            
        return chop
    
    chop = calculate_chop(high_1d, low_1d, close_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in low volatility regime (chop > 50 indicates ranging market)
        if chop_aligned[i] <= 50:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume above average
        volume_ok = volume[i] > volume_ma[i]
        
        if position == 0:
            # Long: price near L4 support with volume confirmation
            if (low[i] <= camarilla_l4_aligned[i] * 1.002 and  # within 0.2% of L4
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short: price near H4 resistance with volume confirmation
            elif (high[i] >= camarilla_h4_aligned[i] * 0.998 and  # within 0.2% of H4
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price reaches opposite H4 level or chop decreases
            if (high[i] >= camarilla_h4_aligned[i] * 0.998 or
                chop_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches opposite L4 level or chop decreases
            if (low[i] <= camarilla_l4_aligned[i] * 1.002 or
                chop_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals