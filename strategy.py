#!/usr/bin/env python3
# 1d_Wick_Reversal_Pivot_Trend
# Hypothesis: At daily timeframe, long wicks (rejections) at pivot levels (R1/S1) indicate exhaustion and reversal. 
# Combined with weekly trend filter to avoid counter-trend trades. Works in both bull and bear markets by 
# capturing reversals at key levels with trend alignment. Low frequency due to strict conditions.

name = "1d_Wick_Reversal_Pivot_Trend"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily pivot levels (standard formula)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot_point = (daily_high + daily_low + daily_close) / 3
    daily_r1 = 2 * pivot_point - daily_low
    daily_s1 = 2 * pivot_point - daily_high
    
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Calculate daily range for wick detection
    daily_range = daily_high - daily_low
    # Avoid division by zero
    daily_range = np.where(daily_range == 0, 1e-10, daily_range)
    
    # Upper wick ratio: (high - close) / range for bearish rejection
    # Lower wick ratio: (close - low) / range for bullish rejection
    upper_wick_ratio = (daily_high - daily_close) / daily_range
    lower_wick_ratio = (daily_close - daily_low) / daily_range
    
    upper_wick_ratio_aligned = align_htf_to_ltf(prices, df_1d, upper_wick_ratio)
    lower_wick_ratio_aligned = align_htf_to_ltf(prices, df_1d, lower_wick_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA34 (34) and daily data (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(daily_r1_aligned[i]) or 
            np.isnan(daily_s1_aligned[i]) or 
            np.isnan(upper_wick_ratio_aligned[i]) or 
            np.isnan(lower_wick_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Wick rejection conditions (strong rejection = ratio > 0.6)
        strong_upper_wick = upper_wick_ratio_aligned[i] > 0.6  # Long upper wick = bearish rejection
        strong_lower_wick = lower_wick_ratio_aligned[i] > 0.6  # Long lower wick = bullish rejection
        
        if position == 0:
            # Long entry: downtrend + price near S1 + strong lower wick (bullish rejection)
            near_s1 = low[i] <= daily_s1_aligned[i] * 1.005  # Within 0.5% of S1
            if downtrend and near_s1 and strong_lower_wick:
                signals[i] = 0.25
                position = 1
            # Short entry: uptrend + price near R1 + strong upper wick (bearish rejection)
            elif uptrend and high[i] >= daily_r1_aligned[i] * 0.995 and strong_upper_wick:  # Within 0.5% of R1
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or wick rejection at resistance
            if not downtrend or upper_wick_ratio_aligned[i] > 0.6:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or wick rejection at support
            if not uptrend or lower_wick_ratio_aligned[i] > 0.6:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals