#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_VolumeSpike
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>2.0x 20-period MA).
Long when price breaks above upper Donchian in 12h uptrend with volume spike. Short when price breaks below lower Donchian in 12h downtrend with volume spike.
Uses discrete position sizing (0.25) to minimize fee churn. Donchian levels calculated from prior 20 periods to avoid look-ahead.
Designed to work in both bull and bear markets by following the 12h trend. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels from prior 20 periods (avoid look-ahead)
    # Using rolling window with shift(1) to use only past data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    uptrend_12h = close > ema_50_12h_aligned
    downtrend_12h = close < ema_50_12h_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA (tight threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian + 1 for shift + 50 for 12h EMA + 20 for volume MA)
    start_idx = 91
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with 12h uptrend and volume spike
            if (close[i] > donchian_upper[i] and 
                uptrend_12h[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with 12h downtrend and volume spike
            elif (close[i] < donchian_lower[i] and 
                  downtrend_12h[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below lower Donchian (breakdown) OR 12h trend changes to downtrend
            if (close[i] < donchian_lower[i] or not uptrend_12h[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above upper Donchian (breakout) OR 12h trend changes to uptrend
            if (close[i] > donchian_upper[i] or not downtrend_12h[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0