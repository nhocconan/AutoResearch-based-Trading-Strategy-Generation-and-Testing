#!/usr/bin/env python3
"""
12h_1d_1w_KAMA_Trend_Filter
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on 12h for trend direction,
with 1d volume confirmation and 1w trend filter to avoid counter-trend trades.
KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends.
Works in both bull and bear markets by only trading in direction of higher timeframe trend.
Target: 15-30 trades/year on 12h (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    # Efficiency Ratio
    er = np.zeros_like(close)
    for i in range(er_length, len(close)):
        if volatility[i-er_length:i+1].sum() > 0:
            er[i] = np.abs(close[i] - close[i-er_length]) / volatility[i-er_length:i+1].sum()
        else:
            er[i] = 0
    
    # Smoothing Constant
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Calculate KAMA on 12h close
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # Calculate 1d volume average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    
    # Align all data to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20.values)
    weekly_bullish = (weekly_close > weekly_open).astype(float)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):  # Wait for KAMA to stabilize
        # Skip if any required data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        # Find corresponding 1d index for current 12h bar
        # Since we aligned the data, we can use the aligned volume MA
        vol_expanded = volume[i] > (vol_ma_aligned[i] * 1.5) if not np.isnan(vol_ma_aligned[i]) else False
        
        # Trend filters
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        
        # Long conditions: price above KAMA, volume expansion, weekly bullish
        if price_above_kama and vol_expanded and weekly_bull:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        # Short conditions: price below KAMA, volume expansion, weekly bearish
        elif price_below_kama and vol_expanded and not weekly_bull:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        # Exit conditions: trend change or loss of volume confirmation
        else:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_KAMA_Trend_Filter"
timeframe = "12h"
leverage = 1.0