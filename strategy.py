#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRVolFilter_Trend
Hypothesis: 4h Donchian(20) breakout with ATR-based volume filter and 1d EMA50 trend filter.
Long when price breaks above upper Donchian channel with volume > 1.5x ATR-scaled MA and 1d uptrend.
Short when price breaks below lower Donchian channel with volume > 1.5x ATR-scaled MA and 1d downtrend.
Uses ATR-scaled volume threshold to adapt to volatility regimes, reducing false breakouts in choppy markets.
Discrete position sizing (0.25) to minimize fee churn. Designed for 20-50 trades/year on 4h timeframe.
Works in both bull and bear markets by following the 1d trend. ATR filter prevents entries during low-volume consolidations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Calculate ATR(14) for volume filter scaling
    tr1 = pd.Series(high).shift(1) - pd.Series(low).shift(1)
    tr2 = abs(pd.Series(high).shift(1) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low).shift(1) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) scaled by ATR for dynamic threshold
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * (1.0 + atr / close * 10)  # Scale threshold by volatility (ATR/price)
    volume_filter = volume > vol_threshold
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 14 for ATR, 20 for vol MA, 50 for EMA)
    start_idx = max(20, 14, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume filter and 1d uptrend
            if (close[i] > donchian_upper[i] and 
                volume_filter[i] and uptrend_1d[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume filter and 1d downtrend
            elif (close[i] < donchian_lower[i] and 
                  volume_filter[i] and downtrend_1d[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below lower Donchian (breakdown) OR 1d trend changes to downtrend
            if (close[i] < donchian_lower[i] or not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above upper Donchian (breakout) OR 1d trend changes to uptrend
            if (close[i] > donchian_upper[i] or not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_ATRVolFilter_Trend"
timeframe = "4h"
leverage = 1.0