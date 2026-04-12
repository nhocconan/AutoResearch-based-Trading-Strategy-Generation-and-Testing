#!/usr/bin/env python3
"""
6h_1w_1d_ElderRay_Power_With_Regime
Hypothesis: Elder Ray Index (bull/bear power) from 1-day data combined with 1-week trend filter. 
Long when 1-day bull power > 0 AND 1-week close > 1-week SMA20; short when 1-day bear power < 0 AND 1-week close < 1-week SMA20.
Uses volume confirmation (>1.5x 30-period average) to filter weak breakouts.
Designed for 6h timeframe to capture medium-term momentum in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_ElderRay_Power_With_Regime"
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
    
    # === 1-DAY ELDER RAY (BULL/BEAR POWER) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:  # need enough for EMA13
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 of 1-day close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13_1d
    
    # Align to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === 1-WEEK TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 20-period SMA of weekly close
    sma20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma20_1w_6h = align_htf_to_ltf(prices, df_1w, sma20_1w)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(sma20_1w_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long conditions: bull power positive, weekly close above weekly SMA20, volume confirmation
        long_signal = (bull_power_6h[i] > 0) and (close_1w[-1] > sma20_1w[-1] if len(close_1w) > 0 else False) and (vol_ratio[i] > 1.5)
        # Simplify weekly close check: use aligned weekly close vs SMA
        # Get aligned weekly close for current 6h bar
        df_1w_temp = get_htf_data(prices, '1w')
        if len(df_1w_temp) >= 2:
            close_1w_arr = df_1w_temp['close'].values
            close_1w_aligned = align_htf_to_ltf(prices, df_1w_temp, close_1w_arr)
            weekly_close_above_sma = close_1w_aligned[i] > sma20_1w_6h[i]
        else:
            weekly_close_above_sma = False
        
        long_signal = (bull_power_6h[i] > 0) and weekly_close_above_sma and (vol_ratio[i] > 1.5)
        
        # Short conditions: bear power negative, weekly close below weekly SMA20, volume confirmation
        weekly_close_below_sma = close_1w_aligned[i] < sma20_1w_6h[i] if len(df_1w_temp) >= 2 else False
        short_signal = (bear_power_6h[i] < 0) and weekly_close_below_sma and (vol_ratio[i] > 1.5)
        
        # Exit conditions: opposite Elder Ray signal
        exit_long = (position == 1) and (bear_power_6h[i] < 0)
        exit_short = (position == -1) and (bull_power_6h[i] > 0)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals