#!/usr/bin/env python3
"""
1h_BB_Reversal_4H_Trend_1D_Volume
Hypothesis: On 1h, trade mean-reversion at Bollinger Bands (±2σ) in the direction of 4h EMA50 trend, confirmed by 1d volume surge. This captures pullbacks in trending markets, avoiding whipsaw. 4h/1d filters reduce trades to 15-30/year by ensuring alignment with higher timeframe momentum. Works in bull/bear via trend filter.
"""

name = "1h_BB_Reversal_4H_Trend_1D_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 1h data for Bollinger Bands
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Bollinger Bands (20) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(sma20[i]) or np.isnan(std20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 4h EMA50
        uptrend_4h = close[i] > ema50_4h_aligned[i]
        downtrend_4h = close[i] < ema50_4h_aligned[i]
        
        # Volume filter: current 1h volume > 1.5x 1d 20-period MA
        volume_filter = volume[i] > vol_ma20_1d_aligned[i] * 1.5
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: price touches lower BB in uptrend with volume
            if close[i] <= lower[i] and uptrend_4h and volume_filter and in_session:
                signals[i] = 0.20
                position = 1
            # Short: price touches upper BB in downtrend with volume
            elif close[i] >= upper[i] and downtrend_4h and volume_filter and in_session:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price touches middle BB or trend fails
            if close[i] >= sma20[i] or not uptrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price touches middle BB or trend fails
            if close[i] <= sma20[i] or not downtrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals