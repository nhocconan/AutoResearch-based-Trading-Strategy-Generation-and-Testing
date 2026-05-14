#!/usr/bin/env python3
# 4h_Keltner_Donchian_Breakout_1dTrend
# Hypothesis: Combine Keltner Channel breakout with Donchian trend filter on 1d for high-probability trend entries.
# Keltner (ATR-based) captures volatility breakouts, while 1d Donchian ensures alignment with higher timeframe trend.
# Works in bull markets via breakouts and in bear via mean-reversion off bands when trend reverses.
# Target: 20-30 trades/year with low turnover to minimize fee drag.

name = "4h_Keltner_Donchian_Breakout_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 2.0)
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    upper = ma + 2.0 * atr
    lower = ma - 2.0 * atr
    
    # Donchian Channel (20) on 1d for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close above Keltner upper AND price above 1d Donchian low (uptrend bias)
            if close[i] > upper[i] and close[i] > donchian_low_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below Keltner lower AND price below 1d Donchian high (downtrend bias)
            elif close[i] < lower[i] and close[i] < donchian_high_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below Keltner middle OR trend bias lost
            if close[i] < ma[i] or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above Keltner middle OR trend bias lost
            if close[i] > ma[i] or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals