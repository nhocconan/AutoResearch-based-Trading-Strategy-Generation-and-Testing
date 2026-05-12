#!/usr/bin/env python3
"""
6h_1d_1w_TurtleSqueeze_Breakout
Hypothesis: Combine 1-day Donchian(20) breakout with 1-week volatility squeeze (BB width < 20th percentile) and volume confirmation on 6h timeframe.
Only long when price breaks above 1d upper band with volatility squeeze and volume spike.
Only short when price breaks below 1d lower band with volatility squeeze and volume spike.
Designed to catch explosive moves after low volatility periods, works in both bull and bear markets by filtering for volatility contraction before expansion.
"""

name = "6h_1d_1w_TurtleSqueeze_Breakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Volume spike: >2.0x 30-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Donchian channels (20-period)
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # 1w data for volatility squeeze (Bollinger Bands width)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-week Bollinger Bands (20, 2.0)
    bb_middle = pd.Series(df_1w['close']).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(df_1w['close']).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate 50-period percentile of BB width for squeeze detection (width < 20th percentile = squeeze)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 20 else np.nan, raw=False
    ).values
    volatility_squeeze = bb_width_percentile < 0.20  # 20th percentile
    
    # Align volatility squeeze to 6h timeframe
    volatility_squeeze_aligned = align_htf_to_ltf(prices, df_1w, volatility_squeeze)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(volatility_squeeze_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 1d Donchian high + volatility squeeze + volume spike
            if (close[i] > donch_high_aligned[i] and 
                volatility_squeeze_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 1d Donchian low + volatility squeeze + volume spike
            elif (close[i] < donch_low_aligned[i] and 
                  volatility_squeeze_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters 1d Donchian channel OR volatility expansion (end of squeeze)
            if (close[i] > donch_low_aligned[i] and close[i] < donch_high_aligned[i]) or \
               (~volatility_squeeze_aligned[i]):  # volatility expansion
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters 1d Donchian channel OR volatility expansion
            if (close[i] > donch_low_aligned[i] and close[i] < donch_high_aligned[i]) or \
               (~volatility_squeeze_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals