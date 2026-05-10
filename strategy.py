#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R1, S1) on 1d chart identify key intraday support/resistance.
# Breakout above R1 with 1d uptrend (price > EMA34) and volume confirmation signals bullish momentum.
# Breakdown below S1 with 1d downtrend (price < EMA34) and volume confirmation signals bearish momentum.
# Uses 1d timeframe for trend and pivot levels to avoid noise, 4h for execution.
# Designed for low trade frequency (20-40/year) to minimize fee drift.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Fetch 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels for 1d: R1, S1
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    camarilla_width = 1.1 * (high_1d - low_1d) / 12
    r1_1d = close_1d_arr + camarilla_width
    s1_1d = close_1d_arr - camarilla_width
    
    # Align 1d indicators to 4h timeframe (wait for close of 1d bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 4h EMA50 for entry filter (optional trend alignment)
    close_series = pd.Series(close)
    ema50_4h = close_series.ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34) + 5  # Enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or \
           np.isnan(s1_1d_aligned[i]) or np.isnan(ema50_4h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend condition: 1d EMA34 slope (use level as proxy)
        uptrend_1d = close > ema34_1d_aligned[i]
        downtrend_1d = close < ema34_1d_aligned[i]
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long breakout: price crosses above R1 with uptrend and volume
            if close[i] > r1_1d_aligned[i] and uptrend_1d and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price crosses below S1 with downtrend and volume
            elif close[i] < s1_1d_aligned[i] and downtrend_1d and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA50 or trend reverses
            if close[i] < ema50_4h[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above EMA50 or trend reverses
            if close[i] > ema50_4h[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals