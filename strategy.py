#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d volume regime filter.
    # Long when price breaks above 20-period 6h high AND 1d volume is in expansion regime (ATR ratio > 1.2).
    # Short when price breaks below 20-period 6h low AND 1d volume is in expansion regime.
    # Exit on opposite Donchian(10) break or volume contraction.
    # Uses discrete size 0.25 to minimize fee churn.
    # Target: 50-150 total trades over 4 years (12-37/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volume regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) with min_periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 20-period ATR mean (expansion > 1.2, contraction < 0.8)
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / np.where(atr_ma_20 > 0, atr_ma_20, 1e-10)
    
    # Align HTF indicators to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 6h Donchian channels (20 for entry, 10 for exit)
    # Donchian(20) high/low for breakout entry
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    # Donchian(10) high/low for exit
    donchian_high_10 = high_series.rolling(window=10, min_periods=10).max().values
    donchian_low_10 = low_series.rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime filters
        volume_expansion = atr_ratio_aligned[i] > 1.2  # ATR expansion regime
        volume_contraction = atr_ratio_aligned[i] < 0.8  # ATR contraction regime
        
        # Entry conditions: Donchian(20) break with volume expansion
        long_entry = (close[i] > donchian_high_20[i] and volume_expansion)
        short_entry = (close[i] < donchian_low_20[i] and volume_expansion)
        
        # Exit conditions: Donchian(10) break in opposite direction OR volume contraction
        long_exit = (close[i] < donchian_low_10[i]) or volume_contraction
        short_exit = (close[i] > donchian_high_10[i]) or volume_contraction
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_donchian_breakout_volume_regime_v1"
timeframe = "6h"
leverage = 1.0