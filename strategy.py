#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-week SuperTrend (ATR-based trend) + 1-day Williams %R for overbought/oversold signals.
# Long when price is above SuperTrend (uptrend) AND Williams %R < -80 (oversold).
# Short when price is below SuperTrend (downtrend) AND Williams %R > -20 (overbought).
# Exit when price crosses SuperTrend (trend reversal) or Williams %R returns to neutral (-50).
# SuperTrend provides robust trend filtering; Williams %R captures mean reversion within trend.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE for SuperTrend calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough for ATR(10) and SuperTrend
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR (10) for SuperTrend
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.full_like(tr, np.nan)
    atr[9] = np.nanmean(tr[1:11])  # First ATR: simple average of first 11 TR
    for i in range(11, len(tr)):
        atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # Calculate SuperTrend (10, 3.0)
    # Basic Upper Band = (high + low)/2 + multiplier * ATR
    # Basic Lower Band = (high + low)/2 - multiplier * ATR
    basic_ub = (high_1w + low_1w) / 2 + 3.0 * atr
    basic_lb = (high_1w + low_1w) / 2 - 3.0 * atr
    
    # Final Upper Band
    final_ub = np.full_like(basic_ub, np.nan)
    final_lb = np.full_like(basic_ub, np.nan)
    for i in range(len(close_1w)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_1w[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close_1w[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # SuperTrend direction
    supertrend = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i == 0:
            supertrend[i] = final_ub[i]
        else:
            if supertrend[i-1] == final_ub[i-1] and close_1w[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            elif supertrend[i-1] == final_ub[i-1] and close_1w[i] > final_ub[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close_1w[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close_1w[i] < final_lb[i]:
                supertrend[i] = final_ub[i]
    
    # Load 1d data ONCE for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need enough for Williams %R
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Align indicators to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 14)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for entries: trend + extreme Williams %R
            # Long: price above SuperTrend (uptrend) AND Williams %R < -80 (oversold)
            if (close[i] > supertrend_aligned[i] and 
                williams_r_aligned[i] < -80):
                position = 1
                signals[i] = position_size
            # Short: price below SuperTrend (downtrend) AND Williams %R > -20 (overbought)
            elif (close[i] < supertrend_aligned[i] and 
                  williams_r_aligned[i] > -20):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below SuperTrend or Williams %R returns to neutral
            if (close[i] <= supertrend_aligned[i] or 
                williams_r_aligned[i] >= -50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above SuperTrend or Williams %R returns to neutral
            if (close[i] >= supertrend_aligned[i] or 
                williams_r_aligned[i] <= -50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1w_SuperTrend_1d_WilliamsR_v1"
timeframe = "4h"
leverage = 1.0