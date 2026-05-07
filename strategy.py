#!/usr/bin/env python3
# 1d_1w_Camarilla_Pivot_Trend_Volume
# Daily Camarilla pivot levels with weekly trend filter and volume confirmation.
# Long when price breaks above R1 in weekly uptrend with volume spike, short when breaks below S1 in weekly downtrend.
# Uses weekly EMA20 for trend filter and volume spike (2x 20-day average).
# Designed for low trade frequency to avoid fee drag, targeting 20-50 trades per year.
# Works in both bull and bear markets by following weekly trend.

name = "1d_1w_Camarilla_Pivot_Trend_Volume"
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
    
    # Get weekly data for Camarilla pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    # Shift by 1 to use previous week's data (no look-ahead)
    high_1w = df_1w['high'].shift(1).values
    low_1w = df_1w['low'].shift(1).values
    close_1w = df_1w['close'].shift(1).values
    
    # Calculate pivot and levels
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1 = pivot + (range_1w * 1.1 / 12)
    s1 = pivot - (range_1w * 1.1 / 12)
    
    # Align weekly levels to daily timeframe
    r1_daily = align_htf_to_ltf(prices, df_1w, r1)
    s1_daily = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_daily = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily volume filter (20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_daily[i]) or np.isnan(s1_daily[i]) or 
            np.isnan(ema_20_daily[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above R1 with weekly uptrend and volume spike
            if close[i] > r1_daily[i] and close[i] > ema_20_daily[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S1 with weekly downtrend and volume spike
            elif close[i] < s1_daily[i] and close[i] < ema_20_daily[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price returns below EMA20 or breaks below S1 (with min hold)
            if bars_since_entry >= 2 and (close[i] < ema_20_daily[i] or close[i] < s1_daily[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above EMA20 or breaks above R1 (with min hold)
            if bars_since_entry >= 2 and (close[i] > ema_20_daily[i] or close[i] > r1_daily[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals