#!/usr/bin/env python3
# 12h_KAMA_Direction_With_Trend_Filter
# Hypothesis: KAMA adapts to market noise, providing a smooth trend line that reduces whipsaw.
# In trending markets, price stays near KAMA; in ranging markets, it oscillates around it.
# We go long when price is above KAMA with upward slope and volume confirmation,
# short when below KAMA with downward slope and volume confirmation.
# Uses 1-week EMA34 as higher timeframe trend filter to avoid counter-trend trades.
# Designed for low trade frequency (15-25/year) on 12h timeframe to minimize fee drag.

name = "12h_KAMA_Direction_With_Trend_Filter"
timeframe = "12h"
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
    
    # Get 1-week data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend filter (smooth, lag-appropriate)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA on 12h data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum |diff| over 10 periods
    # Avoid division by zero
    er = np.zeros_like(close)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    # Smoothing constants
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Slope of KAMA (1-period change)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # Volume confirmation (20-period average on 12h = ~10 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 34) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]) or \
           np.isnan(kama_slope[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirm = volume[i] > 1.3 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price above KAMA, KAMA rising, with volume confirmation and weekly uptrend
            if close[i] > kama[i] and kama_slope[i] > 0 and \
               volume_confirm and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, KAMA falling, with volume confirmation and weekly downtrend
            elif close[i] < kama[i] and kama_slope[i] < 0 and \
                 volume_confirm and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA OR KAMA slope turns negative
            if close[i] < kama[i] or kama_slope[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA OR KAMA slope turns positive
            if close[i] > kama[i] or kama_slope[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals