#!/usr/bin/env python3
"""
4h_1d_Donchian_Breakout_With_Volume_Confirmation
Hypothesis: 4-hour Donchian channel breakouts with volume confirmation and 1-day trend filter capture high-probability swing moves in both bull and bear markets.
Breakouts occur when price closes above/below the 20-period 4h Donchian channel with volume > 1.5x 20-period 4h average, filtered by 1-day EMA trend.
Targets 20-30 trades/year per symbol (~80-120 total over 4 years) to minimize fee drag.
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
    
    # 4h Donchian channel (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # 1-day EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        ema21_1d = np.full(len(prices), np.nan)
    else:
        close_1d = df_1d['close'].values
        ema21_1d_raw = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema21_1d = align_htf_to_ltf(prices, df_1d, ema21_1d_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # warmup for Donchian
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema21_1d[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: close above Donchian high with volume expansion and 1-day uptrend
        long_signal = (close[i] > donchian_high[i] and 
                      volume_expansion[i] and 
                      close[i] > ema21_1d[i])
        
        # Short signal: close below Donchian low with volume expansion and 1-day downtrend
        short_signal = (close[i] < donchian_low[i] and 
                       volume_expansion[i] and 
                       close[i] < ema21_1d[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Donchian_Breakout_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0