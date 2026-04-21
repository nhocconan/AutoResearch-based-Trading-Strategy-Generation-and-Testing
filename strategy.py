#!/usr/bin/env python3
"""
12h_HTF_1w_Donchian_Breakout_VolumeRegime_V1
Hypothesis: Use 1w Donchian(20) breakout with volume confirmation (>1.5x 20-bar MA) and 1d chop regime filter (CHOP > 61.8 = range, < 38.2 = trending). Enter long on Donchian high break in trending/regime, short on low break. ATR-based stoploss (2.0x). Target 12-37 trades/year on 12h timeframe. Works in bull via breakouts, bears via short breakdowns and regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')  # for Donchian channels
    df_1d = get_htf_data(prices, '1d')  # for chop regime filter
    
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1w Donchian Channels (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = hh - ll
    denominator[denominator == 0] = 1e-10
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum / denominator) / log10(14)
    chop = 100 * np.log10(tr_sum / denominator) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    # For breakout strategy, we want trending markets (CHOP < 38.2)
    trending_regime = chop_aligned < 38.2
    
    # === 12h Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume MA (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        in_trending_regime = trending_regime[i]  # only trade in trending markets
        
        if position == 0:
            # Long: break above 1w Donchian high with volume and trending regime
            if price > donchian_high_aligned[i] and vol_ok and in_trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: break below 1w Donchian low with volume and trending regime
            elif price < donchian_low_aligned[i] and vol_ok and in_trending_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < close[i-1] - 2.0 * atr[i] or (price < donchian_low_aligned[i] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > close[i-1] + 2.0 * atr[i] or (price > donchian_high_aligned[i] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_1w_Donchian_Breakout_VolumeRegime_V1"
timeframe = "12h"
leverage = 1.0