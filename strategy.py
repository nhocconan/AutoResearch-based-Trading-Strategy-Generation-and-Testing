#!/usr/bin/env python3
# 4H_1D_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: Use 1d Camarilla R1/S1 levels for breakout entries on 4h, filtered by 12h EMA trend and volume confirmation.
# Camarilla levels provide institutional pivot points with proven mean-reversion/breakout behavior.
# Works in bull/bear by filtering with 12h EMA trend, reducing whipsaws in choppy markets.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years).

name = "4H_1D_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1 = close_1d + camarilla_range
    s1 = close_1d - camarilla_range
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA-50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume average (20-period) for volume confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg)  # Use 12h vol avg for stability
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > (vol_avg_aligned[i] * 1.5)
        
        if position == 0:
            # Enter long: price breaks above R1 + 12h EMA uptrend + volume confirmation
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + 12h EMA downtrend + volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S1 or 12h EMA turns down
            if (close[i] < s1_aligned[i] or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R1 or 12h EMA turns up
            if (close[i] > r1_aligned[i] or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals