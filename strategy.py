#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_MACD_Trend
Hypothesis: Combine Elder Ray (Bull/Bear Power) with Zero-Lag MACD for early trend detection on 6h timeframe.
Uses 1d EMA13 as trend filter to avoid counter-trend trades. Works in bull/bear via adaptive trend following.
Target: 12-30 trades/year per symbol (~50-120 total over 4 years) to minimize fee drag.
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
    
    # Get 6h data for indicators
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA13 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for trend filter
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray on 6h
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # EMA13 of 6h close for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_6h - ema13_6h
    # Bear Power = Low - EMA13
    bear_power = low_6h - ema13_6h
    
    # Smooth Bull/Bear Power with EMA8
    bull_power_smooth = pd.Series(bull_power).ewm(span=8, min_periods=8, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=8, min_periods=8, adjust=False).mean().values
    
    # Calculate Zero-Lag MACD on 6h
    # EMA of close
    ema_close = pd.Series(close_6h).ewm(span=34, min_periods=34, adjust=False).mean().values
    # Zero-lag EMA: 2*EMA - EMA(EMA)
    ema_close_lag = pd.Series(ema_close).ewm(span=34, min_periods=34, adjust=False).mean().values
    zl_ema = 2 * ema_close - ema_close_lag
    
    # Signal line: EMA of ZL MACD
    signal_line = pd.Series(zl_ema).ewm(span=9, min_periods=9, adjust=False).mean().values
    # MACD histogram
    macd_hist = zl_ema - signal_line
    
    # Align all 6h indicators to 6h timeframe (they're already aligned)
    bull_power_smooth_aligned = bull_power_smooth
    bear_power_smooth_aligned = bear_power_smooth
    macd_hist_aligned = macd_hist
    
    # Align 1d EMA13
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA13 (13*2 for smoothing), ZL MACD (34+9)
    start_idx = max(50, 34+9)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_smooth_aligned[i]) or 
            np.isnan(bear_power_smooth_aligned[i]) or
            np.isnan(macd_hist_aligned[i]) or
            np.isnan(ema_13_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 1d EMA13
        price_above_ema = close[i] > ema_13_1d_aligned[i]
        price_below_ema = close[i] < ema_13_1d_aligned[i]
        
        # Elder Ray signals: look for divergence and momentum
        # Bullish: Bull Power rising AND MACD histogram turning positive
        bull_momentum = (bull_power_smooth_aligned[i] > bull_power_smooth_aligned[i-1] and 
                        macd_hist_aligned[i] > 0 and macd_hist_aligned[i-1] <= 0)
        # Bearish: Bear Power falling AND MACD histogram turning negative
        bear_momentum = (bear_power_smooth_aligned[i] < bear_power_smooth_aligned[i-1] and 
                        macd_hist_aligned[i] < 0 and macd_hist_aligned[i-1] >= 0)
        
        if position == 0:
            # Long: bullish momentum + price above 1d EMA13
            if bull_momentum and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum + price below 1d EMA13
            elif bear_momentum and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: bearish momentum OR price crosses below 1d EMA13
            if bear_momentum or not price_above_ema:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: bullish momentum OR price crosses above 1d EMA13
            if bull_momentum or not price_below_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroLag_MACD_Trend"
timeframe = "6h"
leverage = 1.0