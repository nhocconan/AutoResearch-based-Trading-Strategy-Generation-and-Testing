#!/usr/bin/env python3

"""
Hypothesis: 12-hour Exponential Moving Average (EMA) crossover with 1-day/1-week trend filter and volume confirmation.
Buy when 12h EMA20 crosses above EMA50, 1d trend is up (price > EMA50), and volume exceeds 1.5x 20-period average.
Sell when 12h EMA20 crosses below EMA50, 1d trend is down (price < EMA50), and volume exceeds 1.5x 20-period average.
Uses 1-week EMA50 as higher timeframe trend filter to avoid counter-trend trades.
Designed for low trade frequency (12-37/year) by requiring EMA crossover + trend alignment + volume spike.
Works in both bull and bear markets by following the 1-day trend while using 1-week for higher timeframe confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA20 and EMA50
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1-week EMA50 for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # EMA crossover signals
        ema20_cross_above = ema20[i] > ema50[i] and ema20[i-1] <= ema50[i-1]
        ema20_cross_below = ema20[i] < ema50[i] and ema20[i-1] >= ema50[i-1]
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: EMA20 crosses above EMA50 + 1d uptrend (price > EMA50_1d) + 1w uptrend + volume spike
            if ema20_cross_above and close[i] > ema50_1d_aligned[i] and ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: EMA20 crosses below EMA50 + 1d downtrend (price < EMA50_1d) + 1w downtrend + volume spike
            elif ema20_cross_below and close[i] < ema50_1d_aligned[i] and ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite EMA crossover or loss of 1d trend
            exit_signal = False
            
            if position == 1:
                # Exit long: EMA20 crosses below EMA50 or price falls below 1d EMA50
                if ema20_cross_below or close[i] < ema50_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: EMA20 crosses above EMA50 or price rises above 1d EMA50
                if ema20_cross_above or close[i] > ema50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_EMA20_50_Crossover_1dTrend_1wFilter_Volume"
timeframe = "12h"
leverage = 1.0