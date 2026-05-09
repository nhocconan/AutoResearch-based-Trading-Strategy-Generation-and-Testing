#!/usr/bin/env python3
name = "12H_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA100 for trend filter
    ema100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 1w EMA100 to 12h timeframe
    ema100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema100_1w)
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels from previous day
    # H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        camarilla_h3[i] = prev_close + 1.1 * range_ / 4
        camarilla_l3[i] = prev_close - 1.1 * range_ / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 12h volume for volume confirmation
    # Use 12h volume EMA20
    volume_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(100, 30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema100_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above 1w EMA100
        uptrend = close[i] > ema100_1w_aligned[i]
        # Downtrend: price below 1w EMA100
        downtrend = close[i] < ema100_1w_aligned[i]
        # Volume surge: current volume > 1.5x 12h volume EMA20
        volume_surge = volume[i] > volume_ema20[i] * 1.5
        
        if position == 0:
            # Enter long: Uptrend + price breaks above Camarilla H3 + volume surge
            if uptrend and close[i] > camarilla_h3_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below Camarilla L3 + volume surge
            elif downtrend and close[i] < camarilla_l3_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below Camarilla L3
            if not uptrend or close[i] < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above Camarilla H3
            if not downtrend or close[i] > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals