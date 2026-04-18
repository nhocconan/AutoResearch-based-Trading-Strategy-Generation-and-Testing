#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ChoppinessIndex_Trend_Regime_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Choppiness Index on daily timeframe (regime filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR for Choppiness Index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum_atr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    choppiness = np.zeros_like(close_1d)
    mask = (range_14 > 0) & (~np.isnan(range_14)) & (~np.isnan(sum_atr_14))
    choppiness[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    
    # Regime: CHOP > 61.8 = ranging (mean reversion), CHOP < 38.2 = trending
    # We'll use trending regime for breakouts
    trending_regime = choppiness < 38.2
    
    # Align regime to 4h
    trending_regime_aligned = align_htf_to_ltf(prices, df_1d, trending_regime.astype(float))
    
    # Daily Donchian channels for breakout signals
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(trending_regime_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        trending_reg = trending_regime_aligned[i] > 0.5  # boolean
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above Donchian high in trending regime with volume
            if close_val > donch_high and trending_reg and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low in trending regime with volume
            elif close_val < donch_low and trending_reg and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below Donchian low (reversal signal)
            if close_val < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above Donchian high (reversal signal)
            if close_val > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals