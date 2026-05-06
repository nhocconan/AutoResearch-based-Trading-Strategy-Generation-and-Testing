#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses Donchian channels for structure, 1d EMA50 for trend alignment, volume spike for confirmation
# Works in both bull/bear markets: breakouts capture momentum, trend filter reduces whipsaw
# Target: 80-180 total trades over 4 years (20-45/year) to minimize fee drag
# Proven pattern: Donchian breakouts with volume work on BTC/ETH in various regimes

name = "4h_Donchian20_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume spike filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Calculate 4h Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    
    # Align HTF indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian High AND uptrend (price > EMA50) AND volume spike
            if close[i] > donchian_high[i] and close[i] > ema50_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < Donchian Low AND downtrend (price < EMA50) AND volume spike
            elif close[i] < donchian_low[i] and close[i] < ema50_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests Donchian Low from above (trend reversal)
            if close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests Donchian High from below (trend reversal)
            if close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals