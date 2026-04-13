#!/usr/bin/env python3
"""
12h_1d1w_Donchian_Breakout_With_Volume_Confirmation
Hypothesis: 12h Donchian breakout with 1d volume confirmation and 1w trend filter.
Long when price breaks above 12h Donchian upper band + 1d volume > 1.8x 20-period average + 1w close > 1w SMA50.
Short when price breaks below 12h Donchian lower band + 1d volume > 1.8x 20-period average + 1w close < 1w SMA50.
Exit when price crosses 12h Donchian midline or 1w trend reverses.
Designed for 12h timeframe to target 15-30 trades/year with strong trend capture in both bull/bear markets.
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
    
    # 12h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 1d volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_50 = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(sma_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Align 1d volume and check condition
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 1.8)
        
        # 1w trend condition
        uptrend = close[i] > sma_50_aligned[i]
        downtrend = close[i] < sma_50_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_upper[i]
        short_breakout = close[i] < donchian_lower[i]
        
        # Exit conditions
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        trend_reverse_long = close[i] < sma_50_aligned[i]  # uptrend broken
        trend_reverse_short = close[i] > sma_50_aligned[i]  # downtrend broken
        
        if position == 0:
            if long_breakout and vol_condition and uptrend:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition and downtrend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit or trend_reverse_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit or trend_reverse_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d1w_Donchian_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0