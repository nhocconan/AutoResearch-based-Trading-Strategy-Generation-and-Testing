#!/usr/bin/env python3
"""
4h_12h_Medium_Term_Trend_With_Volume_And_Regime_Filter
Hypothesis: Combines 12h EMA trend filter with 4h Donchian breakout and volume confirmation for medium-term trend following.
Uses 12h EMA(50) for trend direction, 4h Donchian(20) breakout for entry timing, volume > 1.5x 20-period average for confirmation,
and avoids choppy markets with 12h Choppiness Index > 61.8. Works in both bull and bear markets by following established trends
with volatility-based exits. Target: 20-50 trades/year on 4h (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h Choppiness Index for regime filter
    atr_period = 14
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    highest_12h = pd.Series(high_12h).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_12h = pd.Series(low_12h).rolling(window=atr_period, min_periods=atr_period).min().values
    
    sum_atr_12h = pd.Series(atr_12h).rolling(window=atr_period, min_periods=atr_period).sum().values
    range_12h = highest_12h - lowest_12h
    
    chop_12h = 100 * np.log10(sum_atr_12h / np.maximum(range_12h, 1e-10)) / np.log10(atr_period)
    chop_12h = np.where(range_12h == 0, 100, chop_12h)
    
    # Get 4h data for entry signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h Donchian(20) channels
    highest_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_expansion_4h = volume_4h > (vol_ma_20_4h * 1.5)
    
    # Align all signals to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    highest_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_4h)
    lowest_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_4h)
    volume_expansion_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_expansion_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(chop_12h_aligned[i]) or
            np.isnan(highest_4h_aligned[i]) or np.isnan(lowest_4h_aligned[i]) or
            np.isnan(volume_expansion_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Regime filter: avoid choppy markets (Choppiness Index > 61.8 = range)
        not_choppy = chop_12h_aligned[i] <= 61.8
        
        # Entry conditions
        long_entry = (uptrend and 
                      close[i] > highest_4h_aligned[i] and 
                      volume_expansion_4h_aligned[i] and
                      not_choppy)
        
        short_entry = (downtrend and 
                       close[i] < lowest_4h_aligned[i] and 
                       volume_expansion_4h_aligned[i] and
                       not_choppy)
        
        # Exit conditions: reverse signal or loss of trend/regime
        long_exit = (not uptrend or not not_choppy)
        short_exit = (not downtrend or not not_choppy)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_12h_Medium_Term_Trend_With_Volume_And_Regime_Filter"
timeframe = "4h"
leverage = 1.0