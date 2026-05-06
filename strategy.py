#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels (R3/S3) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 2.0 * avg_volume(20) on 12h
# Short when price breaks below 1d Camarilla S3 AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 2.0 * avg_volume(20) on 12h
# Exit when price retests the 1d Camarilla pivot point (median of R3/S3)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Camarilla provides strong intraday pivot levels with mean reversion properties
# 1d EMA34 ensures we trade with the dominant daily trend filter
# Volume confirmation validates breakout strength while limiting false signals
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "12h_Camarilla_R3S3_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 completed daily bars for pivot
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    pivot = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3.0
    range_1d = high_1d[-1] - low_1d[-1]
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    midpoint = (r3 + s3) / 2.0
    
    # Create arrays for alignment (same value for all bars within the day)
    r3_array = np.full(len(close_1d), r3)
    s3_array = np.full(len(close_1d), s3)
    midpoint_array = np.full(len(close_1d), midpoint)
    
    # Align 1d Camarilla levels to 12h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_array)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_array)
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint_array)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3, 1d EMA34 > EMA34 previous (uptrend), volume spike, in session
            if (close[i] > r3_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3, 1d EMA34 < EMA34 previous (downtrend), volume spike, in session
            elif (close[i] < s3_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests the 1d Camarilla midpoint
            if close[i] <= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests the 1d Camarilla midpoint
            if close[i] >= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals