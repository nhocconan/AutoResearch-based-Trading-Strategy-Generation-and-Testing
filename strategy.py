#!/usr/bin/env python3
"""
4h_1d_KAMA_Trend_Filter
Hypothesis: Uses Kaufman's Adaptive Moving Average (KAMA) on daily timeframe to determine trend direction,
enters on 4h breakouts aligned with the daily trend, with volume confirmation and ATR-based stop.
Designed to work in both bull and bear markets by following the higher timeframe trend.
Target: 20-40 trades/year on 4h (80-160 total over 4 years).
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
    
    # Get daily data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (10, 2, 30) on daily
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    
    # For first value, avoid NaN
    change[0] = 0
    volatility[0] = 0
    
    # Rolling sum for efficiency ratio
    change_sum = pd.Series(change).rolling(window=10, min_periods=10).sum()
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum()
    
    # Avoid division by zero
    er = np.where(volatility_sum > 0, change_sum / volatility_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Get 4h data for entry signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channel (20) for breakout
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max()
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min()
    
    # Calculate 4h ATR (14) for stop loss and volume filter
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean()
    volume_filter = volume_4h > (vol_ma_20 * 1.5)
    
    # Align all signals to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    highest_20_aligned = align_htf_to_ltf(prices, df_4h, highest_20.values)
    lowest_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_20.values)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14.values)
    volume_filter_aligned = align_htf_to_ltf(prices, df_4h, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    # Track entry price for stop loss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily KAMA
        # Uptrend: price above KAMA, Downtrend: price below KAMA
        daily_trend_long = close_1d[-1] > kama_aligned[i] if len(close_1d) > 0 else False
        daily_trend_short = close_1d[-1] < kama_aligned[i] if len(close_1d) > 0 else False
        
        # Breakout conditions
        breakout_long = close[i] > highest_20_aligned[i]
        breakout_short = close[i] < lowest_20_aligned[i]
        
        # Entry logic: breakout in direction of daily trend with volume confirmation
        if daily_trend_long and breakout_long and volume_filter_aligned[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
                entry_price[i] = close[i]
            else:
                signals[i] = position_size
        elif daily_trend_short and breakout_short and volume_filter_aligned[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
                entry_price[i] = close[i]
            else:
                signals[i] = -position_size
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        
        # Stop loss: 2 * ATR below/above entry
        if position == 1 and not np.isnan(entry_price[i]):
            if close[i] < entry_price[i] - 2.0 * atr_aligned[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1 and not np.isnan(entry_price[i]):
            if close[i] > entry_price[i] + 2.0 * atr_aligned[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_KAMA_Trend_Filter"
timeframe = "4h"
leverage = 1.0