#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Camarilla pivot levels with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 1w Camarilla S3 level AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 2.0 * avg_volume(20) on 6h
# Short when price breaks below 1w Camarilla R3 level AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 2.0 * avg_volume(20) on 6h
# Exit when price reaches the opposite Camarilla level (long exits at R3, short exits at S3)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 1w Camarilla provides strong weekly structural levels with institutional relevance
# 1d EMA34 ensures we trade with the dominant daily trend filter
# Volume confirmation validates breakout strength while limiting false signals
# Works in both bull (buy breakouts above S3) and bear (sell breakdowns below R3) markets

name = "6h_Camarilla1w_S3R3_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least 5 completed weekly bars for pivot calculation
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels
    pivot_1w = (high_1w[-1] + low_1w[-1] + close_1w[-1]) / 3.0
    range_1w = high_1w[-1] - low_1w[-1]
    
    # Camarilla levels (using previous week's data)
    # We need to shift the calculation by 1 to avoid look-ahead
    # Use rolling window to get previous week's OHLC
    if len(high_1w) >= 2:
        # Shift arrays to get previous week's data for current week's calculation
        high_prev = np.roll(high_1w, 1)
        low_prev = np.roll(low_1w, 1)
        close_prev = np.roll(close_1w, 1)
        # Set first element to NaN as there's no previous week
        high_prev[0] = np.nan
        low_prev[0] = np.nan
        close_prev[0] = np.nan
        
        pivot_prev = (high_prev + low_prev + close_prev) / 3.0
        range_prev = high_prev - low_prev
        
        # Calculate Camarilla levels based on previous week
        S3 = pivot_prev - (range_prev * 1.125 / 2)
        S4 = pivot_prev - (range_prev * 1.5)
        R3 = pivot_prev + (range_prev * 1.125 / 2)
        R4 = pivot_prev + (range_prev * 1.5)
    else:
        # Not enough data, return zeros
        return np.zeros(n)
    
    # Align 1w Camarilla levels to 6h timeframe (wait for completed 1w bar)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Camarilla S3, 1d EMA34 > EMA34 previous (uptrend), volume spike, in session
            if (close[i] > S3_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Camarilla R3, 1d EMA34 < EMA34 previous (downtrend), volume spike, in session
            elif (close[i] < R3_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches 1w Camarilla R3 (take profit) or S4 (stop loss)
            if close[i] >= R3_aligned[i] or close[i] <= S4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches 1w Camarilla S3 (take profit) or R4 (stop loss)
            if close[i] <= S3_aligned[i] or close[i] >= R4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals