#!/usr/bin/env python3
# 1d_Stochastic_K_Cross_Overbought
# Hypothesis: The Stochastic Oscillator (%K) identifies overbought/oversold conditions. 
# Long entry when %K crosses above 20 from oversold in a weekly uptrend (price > 200 EMA).
# Short entry when %K crosses below 80 from overbought in a weekly downtrend (price < 200 EMA).
# Uses 1d timeframe with 1h trend filter and volume confirmation to avoid false signals.
# Works in bull markets by buying oversold dips in uptrends and in bear markets by selling overbought rallies in downtrends.
# Volume confirmation (>1.5x 20-period MA) filters low-conviction moves.
# Designed for low trade frequency (<25/year) to minimize fee drag.

name = "1d_Stochastic_K_Cross_Overbought"
timeframe = "1d"
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
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get daily data for Stochastic
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Stochastic %K (14,3,3)
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    k_fast = 100 * (df_1d['close'] - lowest_low) / (highest_high - lowest_low)
    k_fast = np.where((highest_high - lowest_low) == 0, 50, k_fast)  # avoid division by zero
    k = pd.Series(k_fast).rolling(window=3, min_periods=3).mean().values
    k = pd.Series(k).rolling(window=3, min_periods=3).mean().values  # smoothed %K
    
    k_aligned = align_htf_to_ltf(prices, df_1d, k)
    
    # Volume confirmation (20-period MA on 1d)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA200 (200) and Stochastic (14+3+3=20) and volume MA (20)
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(k_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation (>1.5x MA to avoid low-conviction moves)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: %K crosses above 20 from oversold in weekly uptrend + volume
            k_prev = k_aligned[i-1] if i > 0 else 50
            k_curr = k_aligned[i]
            if (k_prev <= 20 and k_curr > 20) and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: %K crosses below 80 from overbought in weekly downtrend + volume
            elif (k_prev >= 80 and k_curr < 80) and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: %K crosses below 80 (overbought) or trend breaks
            k_prev = k_aligned[i-1] if i > 0 else 50
            k_curr = k_aligned[i]
            if (k_prev >= 80 and k_curr < 80) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: %K crosses above 20 (oversold) or trend breaks
            k_prev = k_aligned[i-1] if i > 0 else 50
            k_curr = k_aligned[i]
            if (k_prev <= 20 and k_curr > 20) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals