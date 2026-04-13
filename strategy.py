#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with daily trend filter + volume confirmation
    # Long: price breaks above R4 AND daily close > daily open AND volume > 1.5x avg
    # Short: price breaks below S4 AND daily close < daily open AND volume > 1.5x avg
    # Exit: price retests the pivot point (PP) or opposite Camarilla level (S1/R1)
    # Using 6h timeframe for optimal trade frequency (target 12-37/year), Camarilla pivots from 1d for institutional levels,
    # daily trend filter to align with higher timeframe bias, and volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations: PP = (H+L+C)/3, Range = H-L
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4_1d = pp_1d + range_1d * 1.1 / 2
    r3_1d = pp_1d + range_1d * 1.1 / 4
    r2_1d = pp_1d + range_1d * 1.1 / 6
    r1_1d = pp_1d + range_1d * 1.1 / 12
    s1_1d = pp_1d - range_1d * 1.1 / 12
    s2_1d = pp_1d - range_1d * 1.1 / 6
    s3_1d = pp_1d - range_1d * 1.1 / 4
    s4_1d = pp_1d - range_1d * 1.1 / 2
    
    # Align daily Camarilla levels to 6h
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Daily trend filter: bullish if close > open, bearish if close < open
    daily_bullish = close_1d > open_price  # Note: open_price here is from prices DF, need daily open
    # Fix: get daily open from df_1d
    open_1d = df_1d['open'].values
    daily_bullish = close_1d > open_1d
    daily_bearish = close_1d < open_1d
    
    # Align daily trend to 6h
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    # Get 6h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_r4 = close[i] > r4_1d_aligned[i]
        breakout_s4 = close[i] < s4_1d_aligned[i]
        
        # Retest conditions: price returns to PP or S1/R1
        retest_pp = abs(close[i] - pp_1d_aligned[i]) < (0.001 * pp_1d_aligned[i])  # 0.1% tolerance
        retest_r1 = abs(close[i] - r1_1d_aligned[i]) < (0.001 * r1_1d_aligned[i])
        retest_s1 = abs(close[i] - s1_1d_aligned[i]) < (0.001 * s1_1d_aligned[i])
        
        # Entry logic: Camarilla breakout + daily trend alignment + volume confirmation
        long_entry = breakout_r4 and daily_bullish_aligned[i] and volume_spike[i]
        short_entry = breakout_s4 and daily_bearish_aligned[i] and volume_spike[i]
        
        # Exit logic: retest of PP or opposite Camarilla level
        long_exit = retest_pp or retest_s1 or (close[i] < pp_1d_aligned[i])
        short_exit = retest_pp or retest_r1 or (close[i] > pp_1d_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0