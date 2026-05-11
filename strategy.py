#!/usr/bin/env python3
"""
4h_Donchian_Breakout_20_1dTrend_VolumeFilter
Hypothesis: Trade Donchian(20) breakouts with 1d trend filter and volume confirmation. 
Uses price channel breakouts (proven edge) with trend alignment to work in bull/bear markets. 
Volume filter reduces false breakouts. Target: 20-40 trades/year on 4h.
"""

name = "4h_Donchian_Breakout_20_1dTrend_VolumeFilter"
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
    
    # === Daily OHLC for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Donchian Channel (20-period) ===
    # Calculate rolling high/low for 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Volume Filter (1.5x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Donchian and daily calculations)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with uptrend and volume
            if (close[i] > donchian_high[i] and 
                close[i] > ema34_4h[i] and 
                volume_ok[i]):
                signals[i] = 0.30
                position = 1
            # Short breakdown: price breaks below Donchian low with downtrend and volume
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema34_4h[i] and 
                  volume_ok[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low (reversal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # maintain position
        elif position == -1:
            # Short exit: price breaks above Donchian high (reversal)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # maintain position
    
    return signals