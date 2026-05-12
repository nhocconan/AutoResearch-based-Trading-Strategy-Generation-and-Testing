#!/usr/bin/env python3
# 1D_DONCHIAN_BREAKOUT_WEEKLYTREND_VOLUME_CONFIRMATION
# Hypothesis: On 1d timeframe, enter long when price breaks above weekly Donchian(20) high with volume spike and weekly uptrend (close > weekly EMA20).
# Enter short when price breaks below weekly Donchian(20) low with volume spike and weekly downtrend (close < weekly EMA20).
# Exit when price returns to the opposite Donchian band.
# Weekly trend filter and volume confirmation reduce false breakouts.
# Designed for low trade frequency (7-25/year) to minimize fee drag and work in both bull and bear markets.

name = "1D_DONCHIAN_BREAKOUT_WEEKLYTREND_VOLUME_CONFIRMATION"
timeframe = "1d"
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
    
    # Weekly Donchian channel (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian bands: upper = 20-period high, lower = 20-period low
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Weekly EMA20 for trend filter
    ema20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20)
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 2.0  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema20_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly Donchian high with volume spike and weekly uptrend
            if close[i] > donchian_high_aligned[i] and vol_spike[i] and close[i] > ema20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low with volume spike and weekly downtrend
            elif close[i] < donchian_low_aligned[i] and vol_spike[i] and close[i] < ema20_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals