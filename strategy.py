#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Regime_ATRStop_V2
Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter captures strong directional moves while avoiding choppy markets. ATR-based stoploss manages risk. Works in both bull and bear markets: Donchian breakouts capture momentum regardless of regime, volume confirmation filters false breakouts, and choppiness regime avoids whipsaws in ranging markets. Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for choppiness regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Choppiness Index (14-period) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest High and Lowest Low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CI = 100 * log10(sumTR14 / (HH14 - LL14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chopping_raw = np.where(range_14 > 0, sum_tr_14 / range_14, 1.0)
    chopping_raw = np.maximum(chopping_raw, 1e-10)  # prevent log(0)
    chopping_index = 100 * np.log10(chopping_raw) / np.log10(14)
    
    # Align to 4h timeframe
    chopping_index_aligned = align_htf_to_ltf(prices, df_1d, chopping_index)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian Channel (20-period)
    donchian_period = 20
    dc_upper = pd.Series(high_4h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    dc_lower = pd.Series(low_4h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_4h > (1.5 * vol_ma)
    
    # ATR (14-period) for stoploss
    tr1_4h = pd.Series(high_4h - low_4h)
    tr2_4h = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3_4h = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(atr_4h[i]) 
            or np.isnan(chopping_index_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        
        if position == 0:
            # Long: Donchian breakout above upper band + volume spike + trending regime (CHOP < 38.2)
            if (price > dc_upper[i] and volume_spike[i] and 
                chopping_index_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Donchian breakdown below lower band + volume spike + trending regime (CHOP < 38.2)
            elif (price < dc_lower[i] and volume_spike[i] and 
                  chopping_index_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back below lower Donchian band or choppy regime
            elif price < dc_lower[i] or chopping_index_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back above upper Donchian band or choppy regime
            elif price > dc_upper[i] or chopping_index_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Regime_ATRStop_V2"
timeframe = "4h"
leverage = 1.0