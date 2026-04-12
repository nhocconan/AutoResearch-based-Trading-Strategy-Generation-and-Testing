#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_confirmation"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel from 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation: current volume > 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not ready
        if np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donch_high_aligned[i]
        short_breakout = close[i] < donch_low_aligned[i]
        
        # Entry with volume confirmation
        if long_breakout and volume_ok[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and volume_ok[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit on opposite breakout
        elif short_breakout and position == 1:
            position = 0
            signals[i] = 0.0
        elif long_breakout and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h Donchian(20) breakout with 1d channel context and volume confirmation
# Works in bull/bear markets by capturing breakouts in direction of higher timeframe trend
# Volume filter reduces false breakouts, keeping trades ~25-40/year to avoid fee drag
# Position size 0.25 limits drawdown during adverse moves (e.g., 2022 crash)