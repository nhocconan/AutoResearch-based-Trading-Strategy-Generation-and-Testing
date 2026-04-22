#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-week trend filter and volume confirmation.
Long when price breaks above 20-period Donchian upper band and 1-week EMA50 rising with volume spike.
Short when price breaks below 20-period Donchian lower band and 1-week EMA50 falling with volume spike.
Exit when price crosses the midline or 1-week EMA50 reverses.
Donchian channels capture breakouts with clear levels; 1-week EMA provides higher-timeframe trend filter;
volume spike confirms institutional participation. Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 1-week trend.
"""

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
    
    # Donchian channel: 20-period high/low
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    # Middle band: average of upper and lower
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    upper = rolling_max(high, 20)
    lower = rolling_min(low, 20)
    middle = (upper + lower) / 2.0
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1-week close for trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above upper band and 1-week EMA50 rising with volume spike
            if close[i] > upper[i] and ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band and 1-week EMA50 falling with volume spike
            elif close[i] < lower[i] and ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses midline or 1-week EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle or 1-week EMA50 turns down
                if close[i] < middle[i] or ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle or 1-week EMA50 turns up
                if close[i] > middle[i] or ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_DonchianBreakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0