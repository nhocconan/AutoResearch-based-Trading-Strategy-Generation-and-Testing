#!/usr/bin/env python3
"""
1d_ema_cross_volume_1w_filter_v1
Hypothesis: Daily EMA crossover (50/200) with volume confirmation and weekly trend filter.
Works in both bull and bear markets by using weekly EMA to determine long-term trend direction.
Only takes longs in weekly uptrend, shorts in weekly downtrend. Volume confirms breakout strength.
Target: 15-25 trades/year to minimize fee drag while capturing major trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema_cross_volume_1w_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMAs for crossover signal
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_200 = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMAs to daily timeframe
    weekly_ema_50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_50)
    weekly_ema_200_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_200)
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or 
            np.isnan(weekly_ema_50_aligned[i]) or np.isnan(weekly_ema_200_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Daily EMA crossover signals
        ema_bullish = ema_50[i] > ema_200[i]
        ema_bearish = ema_50[i] < ema_200[i]
        
        # Weekly trend filter
        weekly_uptrend = weekly_ema_50_aligned[i] > weekly_ema_200_aligned[i]
        weekly_downtrend = weekly_ema_50_aligned[i] < weekly_ema_200_aligned[i]
        
        if position == 1:  # Long position
            # Exit: daily EMA turns bearish OR weekly trend turns down
            if not ema_bullish or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: daily EMA turns bullish OR weekly trend turns up
            if not ema_bearish or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: daily EMA turns bullish with volume and weekly uptrend
            if ema_bullish and vol_confirm and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: daily EMA turns bearish with volume and weekly downtrend
            elif ema_bearish and vol_confirm and weekly_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals