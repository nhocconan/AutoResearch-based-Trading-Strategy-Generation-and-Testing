#!/usr/bin/env python3
"""
4h_Weekly_Pivot_Breakout_Trend_Filter
Hypothesis: Weekly pivot points provide strong support/resistance. In bull markets (price above weekly pivot), buy breakouts above R1 with volume confirmation and price above daily EMA50. In bear markets (price below weekly pivot), sell breakdowns below S1 with volume confirmation and price below daily EMA50. Uses 4h timeframe for execution with 1d trend filter. Target: 20-40 trades per year (80-160 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Weekly_Pivot_Breakout_Trend_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week's OHLC
    # Weekly high = max of high over last 7 days
    # Weekly low = min of low over last 7 days
    # Weekly close = close of 7 days ago (Friday's close)
    weekly_high = pd.Series(df_1d['high']).rolling(window=7, min_periods=7).max().shift(1).values
    weekly_low = pd.Series(df_1d['low']).rolling(window=7, min_periods=7).min().shift(1).values
    weekly_close = df_1d['close'].shift(7).values
    
    # Weekly pivot point = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly R1 = (2 * P) - L
    weekly_r1 = (2 * weekly_pivot) - weekly_low
    # Weekly S1 = (2 * P) - H
    weekly_s1 = (2 * weekly_pivot) - weekly_high
    
    # Align weekly levels to 4h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 1-day EMA50
        uptrend_regime = close[i] > ema_50_1d_aligned[i]
        downtrend_regime = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: price breaks above weekly R1 in uptrend regime + volume
            long_entry = (close[i] > weekly_r1_aligned[i]) and uptrend_regime and volume_confirm
            # Short: price breaks below weekly S1 in downtrend regime + volume
            short_entry = (close[i] < weekly_s1_aligned[i]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below weekly pivot or regime changes to downtrend
            if (close[i] < weekly_pivot_aligned[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above weekly pivot or regime changes to uptrend
            if (close[i] > weekly_pivot_aligned[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals