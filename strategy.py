#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + Weekly Trend Filter
Long when price > Alligator jaws (13-period SMMA) with volume > 1.5x average and weekly close > weekly open.
Short when price < Alligator jaws with volume > 1.5x average and weekly close < weekly open.
Exit when price crosses Alligator teeth (8-period SMMA).
Designed for low turnover: ~15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA)"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    sma = np.mean(values[:period])
    result = np.full_like(values, np.nan, dtype=float)
    result[period-1] = sma
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Calculate Williams Alligator (13,8,5 SMMA)
    jaw = smma(close, 13)   # 13-period SMMA (blue line)
    teeth = smma(close, 8)  # 8-period SMMA (red line)
    lips = smma(close, 5)   # 5-period SMMA (green line)
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend: 1 if bullish (close > open), -1 if bearish (close < open)
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Create arrays for alignment
    weekly_bullish_arr = weekly_bullish.astype(float)
    weekly_bearish_arr = weekly_bearish.astype(float)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # Get aligned weekly trend values
        weekly_bull = align_htf_to_ltf(prices, df_1w, weekly_bullish_arr)[i]
        weekly_bear = align_htf_to_ltf(prices, df_1w, weekly_bearish_arr)[i]
        
        if np.isnan(weekly_bull) or np.isnan(weekly_bear):
            continue
        
        if position == 0:
            # Long: Price > jaws, volume spike, weekly bullish
            if close[i] > jaw[i] and volume[i] > vol_ma[i] * 1.5 and weekly_bull > 0.5:
                position = 1
                signals[i] = position_size
            # Short: Price < jaws, volume spike, weekly bearish
            elif close[i] < jaw[i] and volume[i] > vol_ma[i] * 1.5 and weekly_bear > 0.5:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price crosses below teeth (8-period SMMA)
            if close[i] < teeth[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price crosses above teeth (8-period SMMA)
            if close[i] > teeth[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Williams_Alligator_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0