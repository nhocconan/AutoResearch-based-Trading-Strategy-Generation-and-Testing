#!/usr/bin/env python3
"""
4h_1d_4h_Trend_Breakout
Hypothesis: 4h price breaks above/below 4h Donchian(20) with 4h volume expansion and 1d EMA(50) trend filter.
Combines breakout momentum with higher timeframe trend alignment to filter false signals.
Works in bull (breakouts up with uptrend) and bear (breakdowns down with downtrend) markets.
Target: 20-30 trades/year.
"""

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
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_donchian = high_roll.values
    lower_donchian = low_roll.values
    
    # 4h volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(volume_expansion[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above upper Donchian with volume expansion and price above 1d EMA50
        long_condition = (close[i] > upper_donchian[i]) and volume_expansion[i] and (close[i] > ema_50_1d_aligned[i])
        
        # Short: breakdown below lower Donchian with volume expansion and price below 1d EMA50
        short_condition = (close[i] < lower_donchian[i]) and volume_expansion[i] and (close[i] < ema_50_1d_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_4h_Trend_Breakout"
timeframe = "4h"
leverage = 1.0