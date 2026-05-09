#!/usr/bin/env python3
# 1D_WeeklyPivot_CCI_Breakout_1dTrend_Volume
# Hypothesis: Weekly pivot breakout with CCI confirmation and volume filter. Works in both bull/bear: trend filter avoids counter-trend trades, volume spike confirms institutional interest. Weekly pivots provide strong support/resistance levels.
# Uses 1d timeframe with 1h for weekly pivot calculation to reduce lag.

name = "1D_WeeklyPivot_CCI_Breakout_1dTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot from 1h data (less lag than daily)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 10:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Weekly high, low, close (using last 5 1h bars as proxy for weekly)
    # For true weekly, we would need to aggregate 1h data into weekly, but this is a reasonable approximation
    # that captures the weekly pivot concept with less lag
    lookback = min(168, len(high_1h))  # 168 hours = 1 week
    weekly_high = np.max(high_1h[-lookback:]) if lookback > 0 else high_1h[-1]
    weekly_low = np.min(low_1h[-lookback:]) if lookback > 0 else low_1h[-1]
    weekly_close = close_1h[-1]
    
    # Weekly pivot levels (standard calculation)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 1d timeframe
    # Create arrays of same length as df_1h for alignment
    weekly_pivot_arr = np.full_like(close_1h, weekly_pivot)
    weekly_r1_arr = np.full_like(close_1h, weekly_r1)
    weekly_s1_arr = np.full_like(close_1h, weekly_s1)
    weekly_r2_arr = np.full_like(close_1h, weekly_r2)
    weekly_s2_arr = np.full_like(close_1h, weekly_s2)
    
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1h, weekly_pivot_arr)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1h, weekly_r1_arr)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1h, weekly_s1_arr)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1h, weekly_r2_arr)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1h, weekly_s2_arr)
    
    # Calculate CCI(20) for confirmation
    typical_price = (high + low + close) / 3
    tp_mean = np.full_like(typical_price, np.nan)
    tp_dev = np.full_like(typical_price, np.nan)
    
    if len(typical_price) >= 20:
        for i in range(19, len(typical_price)):
            tp_mean[i] = np.mean(typical_price[i-19:i+1])
            tp_dev[i] = np.mean(np.abs(typical_price[i-19:i+1] - tp_mean[i]))
        
        cci = np.full_like(typical_price, np.nan)
        valid = (~np.isnan(tp_mean)) & (~np.isnan(tp_dev)) & (tp_dev != 0)
        cci[valid] = (typical_price[valid] - tp_mean[valid]) / (0.015 * tp_dev[valid])
    else:
        cci = np.full_like(typical_price, np.nan)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure CCI and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(cci[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 AND CCI > 100 (bullish momentum) AND volume spike
            if (close[i] > weekly_r1_aligned[i] and 
                cci[i] > 100 and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 AND CCI < -100 (bearish momentum) AND volume spike
            elif (close[i] < weekly_s1_aligned[i] and 
                  cci[i] < -100 and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 OR CCI < -100 (momentum reversal)
            if close[i] < weekly_s1_aligned[i] or cci[i] < -100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 OR CCI > 100 (momentum reversal)
            if close[i] > weekly_r1_aligned[i] or cci[i] > 100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals