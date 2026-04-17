#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1_S1_Breakout_Volume_1wTrend
Hypothesis: Camarilla pivot levels on 1d provide strong support/resistance. Breaking above R1 or below S1 with volume confirmation and 1w trend alignment captures institutional breakouts. Works in bull markets by catching momentum and in bear markets by avoiding false breaks via 1w trend filter. Targets 15-25 trades/year to minimize fee drag.
"""

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar using previous day's OHLC
    R1_1d = np.zeros(len(close_1d))
    S1_1d = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC (i-1) to calculate today's levels
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        range_prev = high_prev - low_prev
        
        # Camarilla equations
        R1_1d[i] = close_prev + (range_prev * 1.1 / 12)
        S1_1d[i] = close_prev - (range_prev * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Volume confirmation: 20-period average on 12h
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA20 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema20_1w = close_series_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1w EMA to 12h timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20)  # volume MA20, need at least 2 days for pivots
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Breakout conditions
        breakout_long = close[i] > R1_1d_aligned[i]
        breakout_short = close[i] < S1_1d_aligned[i]
        
        # 1w trend filter
        uptrend_1w = close[i] > ema20_1w_aligned[i]
        downtrend_1w = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # Long: break above R1 + volume filter + 1w uptrend
            if breakout_long and volume_filter and uptrend_1w:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + volume filter + 1w downtrend
            elif breakout_short and volume_filter and downtrend_1w:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: break below S1 (reversal signal)
            if breakout_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: break above R1 (reversal signal)
            if breakout_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_R1_S1_Breakout_Volume_1wTrend"
timeframe = "12h"
leverage = 1.0