#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_WeeklyTrend
Hypothesis: Daily price in direction of weekly KAMA trend with volume confirmation.
Kaufman's Adaptive Moving Average adapts to market noise, reducing false signals in sideways markets.
Long when price > weekly KAMA and volume > 1.5x average, short when price < weekly KAMA and volume > 1.5x average.
Designed to work in both bull and bear markets by following the adaptive trend.
Target: 10-25 trades/year to minimize fee drag while capturing sustained trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for KAMA trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA (using close prices)
    close_weekly = df_weekly['close'].values
    # ER (Efficiency Ratio) = |change over 10 periods| / sum of absolute changes over 10 periods
    change = np.abs(np.diff(close_weekly, n=10))  # |close[t] - close[t-10]|
    volatility = np.zeros_like(close_weekly)
    for i in range(10, len(close_weekly)):
        volatility[i] = np.sum(np.abs(np.diff(close_weekly[i-9:i+1])))
    # Avoid division by zero
    er = np.zeros_like(close_weekly)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) ≈ 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_weekly)
    kama[0] = close_weekly[0]
    for i in range(1, len(close_weekly)):
        kama[i] = kama[i-1] + sc[i] * (close_weekly[i] - kama[i-1])
    
    # Align KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_weekly, kama)
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly KAMA
        above_kama = close[i] > kama_aligned[i]
        below_kama = close[i] < kama_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: price in direction of trend with volume
        long_entry = vol_confirm and above_kama
        short_entry = vol_confirm and below_kama
        
        # Exit logic: opposite condition or volume drops
        long_exit = below_kama or not vol_confirm
        short_exit = above_kama or not vol_confirm
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Trend_Filter_WeeklyTrend"
timeframe = "1d"
leverage = 1.0