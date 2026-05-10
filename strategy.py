#!/usr/bin/env python3
# 12h_1w_1d_TRIX_Volume_Regime
# Hypothesis: 12h TRIX signal filtered by 1w trend and volume spike. TRIX captures momentum shifts; 1w trend filters direction; volume confirms breakout strength.
# Designed for low trade frequency (~20-40/year) to minimize fee drag and work in bull/bear markets.

name = "12h_1w_1d_TRIX_Volume_Regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX (12-period) on 12h close
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema3).pct_change(periods=1) * 100  # percentage change
    trix_signal = pd.Series(trix).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX (12+8=20) + EMA50 (50) + volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix_signal[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1w EMA50
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        uptrend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # TRIX signal line crossover
        trix_cross_up = trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]
        trix_cross_down = trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]
        
        if position == 0:
            # Long: TRIX bullish crossover with volume surge and 1w uptrend
            if trix_cross_up and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX bearish crossover with volume surge and 1w downtrend
            elif trix_cross_down and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX bearish crossover OR trend changes
            if trix_cross_down or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX bullish crossover OR trend changes
            if trix_cross_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals