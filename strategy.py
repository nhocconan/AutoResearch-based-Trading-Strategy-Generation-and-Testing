#!/usr/bin/env python3
# 1d_1w_Camarilla_Pivot_R3_S3_Breakout_Trend_Filter_Volume
# Hypothesis: Weekly trend filter + daily Camarilla R3/S3 breakouts with volume confirmation
# Reduces false signals while maintaining trend alignment. Works in both bull and bear markets
# by only trading in direction of weekly trend. Target: 15-25 trades/year.

name = "1d_1w_Camarilla_Pivot_R3_S3_Breakout_Trend_Filter_Volume"
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
    
    # Get weekly data for trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly EMA5 for trend filter
    close_1w = df_1w['close'].values
    ema_5_1w = pd.Series(close_1w).ewm(span=5, adjust=False, min_periods=5).mean().values
    ema_5_1d = align_htf_to_ltf(prices, df_1w, ema_5_1w)
    
    # Calculate weekly Camarilla levels (R3, S3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    r3 = close_1w + (high_1w - low_1w) * 1.12 / 4
    s3 = close_1w - (high_1w - low_1w) * 1.12 / 4
    
    # Calculate weekly volume average (10-period) for volume filter
    volume_1w = df_1w['volume'].values
    vol_ma_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to daily timeframe
    r3_1d = align_htf_to_ltf(prices, df_1w, r3)
    s3_1d = align_htf_to_ltf(prices, df_1w, s3)
    vol_ma_10_1d = align_htf_to_ltf(prices, df_1w, vol_ma_10_1w)
    
    # Calculate volume spike on daily timeframe
    vol_ma_10_1d_calc = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma_10_1d_calc)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(5, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or 
            np.isnan(ema_5_1d[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with uptrend (above weekly EMA5) and volume
            if close[i] > r3_1d[i] and close[i] > ema_5_1d[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with downtrend (below weekly EMA5) and volume
            elif close[i] < s3_1d[i] and close[i] < ema_5_1d[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below weekly EMA5 (trend change)
            if close[i] < ema_5_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above weekly EMA5 (trend change)
            if close[i] > ema_5_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals