#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1d Supertrend and 1w ADX trend filter.
# Supertrend(10, 3) on 1d provides dynamic trend direction and trailing stop.
# 1w ADX(14) > 25 confirms strong trend on higher timeframe.
# Long when Supertrend flips to up and 1w ADX > 25.
# Short when Supertrend flips to down and 1w ADX > 25.
# Exit when Supertrend flips opposite direction.
# Designed to capture strong trends while avoiding whipsaws in weak trends.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE for Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR on 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 10
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Supertrend parameters
    factor = 3.0
    
    # Basic upper and lower bands
    basic_ub = (high_1d + low_1d) / 2 + factor * atr_1d
    basic_lb = (high_1d + low_1d) / 2 - factor * atr_1d
    
    # Final upper and lower bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    
    # Initialize first values
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(close_1d)):
        if close_1d[i-1] <= final_ub[i-1]:
            final_ub[i] = min(basic_ub[i], final_ub[i-1])
        else:
            final_ub[i] = basic_ub[i]
            
        if close_1d[i-1] >= final_lb[i-1]:
            final_lb[i] = max(basic_lb[i], final_lb[i-1])
        else:
            final_lb[i] = basic_lb[i]
    
    # Supertrend direction: 1 for uptrend, -1 for downtrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # Start with uptrend assumption
    
    supertrend[0] = final_ub[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] <= final_ub[i]:
            direction[i] = -1
            supertrend[i] = final_ub[i]
        elif close_1d[i] >= final_lb[i]:
            direction[i] = 1
            supertrend[i] = final_lb[i]
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
    
    # Load 1w data ONCE for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on 1w
    period = 14
    
    # True Range
    tr1_w = np.abs(high_1w[1:] - low_1w[1:])
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w = np.concatenate([[np.nan], tr_w])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) / period
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_wilders = wilders_smoothing(tr_w, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_wilders
    minus_di = 100 * minus_dm_smooth / atr_wilders
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, period)
    
    # Align indicators to lower timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 30)  # Need Supertrend and ADX
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Look for Supertrend direction changes
            # Long: Supertrend flips to up AND strong trend
            if (direction_aligned[i] == 1 and 
                direction_aligned[i-1] == -1 and 
                strong_trend):
                position = 1
                signals[i] = position_size
            # Short: Supertrend flips to down AND strong trend
            elif (direction_aligned[i] == -1 and 
                  direction_aligned[i-1] == 1 and 
                  strong_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Supertrend flips to down
            if direction_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Supertrend flips to up
            if direction_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Supertrend_1wADX_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0