#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_1dEMA34
Hypothesis: 4-hour Donchian(20) breakout filtered by 1-day EMA34 trend and volume spikes (2x 20-period EMA).
Long when price breaks above upper band with uptrend and volume spike; short when breaks below lower band with downtrend and volume spike.
Exit on opposite band touch. Designed for moderate trade frequency (~30-50/year) to balance edge and fee drag.
Works in bull/bear by following higher timeframe trend.
"""

name = "4h_Donchian_Breakout_Volume_Trend_1dEMA34"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 4h Donchian(20) Channel ===
    # Upper band: max(high, 20)
    # Lower band: min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Volume Filter: 2.0x 20-period EMA on 4h ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Donchian20 and daily EMA34)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper band with uptrend and volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with downtrend and volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches or crosses below lower band
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price touches or crosses above upper band
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals