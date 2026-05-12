#!/usr/bin/env python3
# 6h_Stochastic_Pullback_WeeklyTrend
# Hypothesis: Use weekly trend direction (based on weekly SMA50) to filter trades on 6h.
# Enter long when weekly trend is up and price pulls back to 6h VWAP (oversold but within trend).
# Enter short when weekly trend is down and price pulls back to 6h VWAP (overbought but within trend).
# Uses slow stochastic (14,3,3) for oversold/overbought signals within the weekly trend.
# Designed for low frequency (10-30 trades/year) to avoid fee drag. Works in bull (buy pullbacks in uptrend)
# and bear (sell pullbacks in downtrend) with weekly trend filter.

name = "6h_Stochastic_Pullback_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def stochastic(high, low, close, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator (%K and %D).
    Returns %K and %D arrays.
    """
    n = len(high)
    lowest_low = np.zeros(n)
    highest_high = np.zeros(n)
    
    for i in range(n):
        if i < k_period:
            lowest_low[i] = np.min(low[0:i+1])
            highest_high[i] = np.max(high[0:i+1])
        else:
            lowest_low[i] = np.min(low[i-k_period+1:i+1])
            highest_high[i] = np.max(high[i-k_period+1:i+1])
    
    # Avoid division by zero
    diff = highest_high - lowest_low
    k_percent = np.where(diff != 0, 100 * (close - lowest_low) / diff, 0)
    
    # Smooth %K to get %D (simple moving average of %K)
    d_percent = np.zeros(n)
    for i in range(n):
        if i < d_period:
            d_percent[i] = np.mean(k_percent[0:i+1])
        else:
            d_percent[i] = np.mean(k_percent[i-d_period+1:i+1])
    
    return k_percent, d_percent

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
    
    # Weekly SMA50 for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # 6h VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    # Slow Stochastic (14,3,3) on 6h data
    k_percent, d_percent = stochastic(high, low, close, k_period=14, d_period=3)
    
    # Align weekly SMA50 to 6h timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure weekly SMA50 is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(k_percent[i]) or np.isnan(d_percent[i]) or np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close[i] < sma_50_1w_aligned[i]
        
        # Stochastic signals for pullback entries
        stoch_oversold = k_percent[i] < 20 and d_percent[i] < 20
        stoch_overbought = k_percent[i] > 80 and d_percent[i] > 80
        
        # Price near VWAP (within 0.5% for entry)
        vwap_distance = abs(close[i] - vwap[i]) / vwap[i]
        near_vwap = vwap_distance < 0.005
        
        if position == 0:
            # LONG: weekly uptrend + oversold stochastic + price near VWAP (pullback)
            if weekly_uptrend and stoch_oversold and near_vwap:
                signals[i] = 0.25
                position = 1
            # SHORT: weekly downtrend + overbought stochastic + price near VWAP (pullback)
            elif weekly_downtrend and stoch_overbought and near_vwap:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: weekly trend turns down OR stochastic becomes overbought (exit pullback)
            if not weekly_uptrend or (k_percent[i] > 80 and d_percent[i] > 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: weekly trend turns up OR stochastic becomes oversold (exit pullback)
            if not weekly_downtrend or (k_percent[i] < 20 and d_percent[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals