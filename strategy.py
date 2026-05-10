#!/usr/bin/env python3
# 12h_1w_Camarilla_R1_S1_Breakout_Trend_Filter
# Hypothesis: 12h strategy using weekly Camarilla R1/S1 breakouts with 1d trend filter (price > 1d EMA50).
# Enters long on break above weekly R1 in uptrend with volume confirmation, short on break below weekly S1 in downtrend.
# Uses weekly timeframe for structure to reduce trade frequency and avoid overtrading.
# Designed for low trade frequency (12-37/year) to work in both bull and bear markets via trend filter.

name = "12h_1w_Camarilla_R1_S1_Breakout_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Camarilla levels (structure)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate weekly Camarilla levels (R1, S1) from prior week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    r1 = close_1w + 1.1 * (high_1w - low_1w) / 12
    s1 = close_1w - 1.1 * (high_1w - low_1w) / 12
    
    # Align R1 and S1 to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation (1.8x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from weekly EMA50
        bullish_trend = close_1w[i] > ema_50_aligned[i]  # Use weekly close for trend
        bearish_trend = close_1w[i] < ema_50_aligned[i]
        
        # Volume confirmation (1.8x average)
        volume_surge = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above R1 in bullish trend with volume
            if close[i] > r1_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 in bearish trend with volume
            elif close[i] < s1_aligned[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price crosses below weekly EMA50
                if close[i] < ema_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price crosses above weekly EMA50
                if close[i] > ema_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals