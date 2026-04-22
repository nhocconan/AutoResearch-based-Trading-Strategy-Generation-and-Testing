#!/usr/bin/env python3

"""
Hypothesis: 12-hour timeframe strategy using 1-month Donchian channel breakouts with 1-day ATR filter and volume confirmation.
Breakouts above upper Donchian channel (20 periods on 1d) trigger long entries when 1-day ATR is below its 50-period median (low volatility regime) and volume exceeds 1.5x 20-period average.
Breakouts below lower Donchian channel trigger short entries under same conditions.
Exits occur on opposite Donchian channel touch or when ATR expands above 2x its 50-period median.
Designed for low trade frequency (12-37/year) by requiring volatility filter and volume confirmation.
Works in both bull and bear markets by following price breakouts with volatility filtering to avoid false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian channels and ATR - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily Donchian Channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Daily ATR (14-period) for volatility filter
    tr1 = pd.Series(high_1d).rolling(window=2, min_periods=2).max().values - pd.Series(low_1d).rolling(window=2, min_periods=2).min().values
    tr2 = abs(pd.Series(high_1d).shift(1).values - pd.Series(low_1d).values)
    tr3 = abs(pd.Series(high_1d).shift(1).values - pd.Series(close).values)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_median_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).median().values
    atr_median_50_aligned = align_htf_to_ltf(prices, df_1d, atr_median_50)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_median_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current ATR (need current daily ATR aligned)
        # Simplified: use current ATR value from aligned array (we'll compute ATR aligned too)
        # For efficiency, compute ATR aligned array
        if i == 60:  # Compute ATR aligned once
            atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
        
        # Volatility filter: current ATR < 1.5 * median ATR (low volatility regime)
        vol_filter = atr_14_aligned[i] < 1.5 * atr_median_50_aligned[i]
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: break above upper Donchian + low volatility + volume spike
            if close[i] > donchian_high_aligned[i] and vol_filter and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian + low volatility + volume spike
            elif close[i] < donchian_low_aligned[i] and vol_filter and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: touch opposite Donchian band or ATR expansion (high volatility)
            exit_signal = False
            
            if position == 1:
                # Exit long: touch lower Donchian or ATR > 2x median (volatility expansion)
                if close[i] < donchian_low_aligned[i] or atr_14_aligned[i] > 2.0 * atr_median_50_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: touch upper Donchian or ATR > 2x median (volatility expansion)
                if close[i] > donchian_high_aligned[i] or atr_14_aligned[i] > 2.0 * atr_median_50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dATR_Volume_Filter"
timeframe = "12h"
leverage = 1.0