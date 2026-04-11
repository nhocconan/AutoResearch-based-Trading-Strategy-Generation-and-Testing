#!/usr/bin/env python3
# 6h_1w_donchian_breakout_volume_v1
# Strategy: 6h Donchian(20) breakout with 1w high/low filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Breakouts above/below 20-period 6h Donchian channels, filtered by weekly trend
# (price above/below weekly Donchian(10)) and confirmed by volume (>1.5x 20-period average),
# capture sustained moves in both bull and bear markets. Weekly filter ensures alignment with
# higher timeframe trend, reducing false breakouts. Designed for low trade frequency
# (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 60-period Donchian for weekly trend (higher timeframe)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20_1w)
    donch_low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20_1w)
    
    # 6h Donchian(20) for entry signals
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after sufficient data for indicators
        # Skip if any required data is invalid
        if np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or \
           np.isnan(donch_high_20_1w_aligned[i]) or np.isnan(donch_low_20_1w_aligned[i]) or \
           np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Weekly trend filter: price vs weekly Donchian(20)
        # Uptrend: price above weekly Donchian high
        # Downtrend: price below weekly Donchian low
        uptrend = close[i] > donch_high_20_1w_aligned[i]
        downtrend = close[i] < donch_low_20_1w_aligned[i]
        
        # Entry conditions
        # Long: Price breaks above 6h Donchian(20) high AND weekly uptrend AND volume confirmation
        if close[i] > donch_high_20[i] and uptrend and vol_confirm and position != 1:
            # Additional check: ensure breakout is new (not already broken)
            if i == 40 or close[i-1] <= donch_high_20[i-1]:
                position = 1
                signals[i] = 0.25
        # Short: Price breaks below 6h Donchian(20) low AND weekly downtrend AND volume confirmation
        elif close[i] < donch_low_20[i] and downtrend and vol_confirm and position != -1:
            # Additional check: ensure breakdown is new
            if i == 40 or close[i-1] >= donch_low_20[i-1]:
                position = -1
                signals[i] = -0.25
        # Exit: Price returns to the opposite side of the 6h Donchian channel
        elif position == 1 and close[i] < donch_low_20[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high_20[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals