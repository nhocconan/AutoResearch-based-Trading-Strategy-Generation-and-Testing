#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter
# Long when price breaks above 4h Donchian high + 1d ATR ratio (ATR14/ATR50) > 1.2 (expanding volatility)
# Short when price breaks below 4h Donchian low + 1d ATR ratio > 1.2
# Uses discrete position sizing (0.25) to minimize fee churn
# Designed for low trade frequency (20-40/year) to capture volatility expansion breakouts
# Works in both bull (breakouts continue) and bear (breakdowns accelerate) markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channel (20) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # === 1d Indicators: ATR Ratio for Volatility Filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # First TR
    
    # ATR(14) and ATR(50) for volatility ratio
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volatility ratio: ATR(14)/ATR(50) > 1.2 indicates expanding volatility
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 0.0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h upper Donchian
        # 2. Expanding volatility (ATR14/ATR50 > 1.2)
        if (close[i] > donchian_high_aligned[i]) and (atr_ratio_aligned[i] > 1.2):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h lower Donchian
        # 2. Expanding volatility (ATR14/ATR50 > 1.2)
        elif (close[i] < donchian_low_aligned[i]) and (atr_ratio_aligned[i] > 1.2):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_ATR_Volatility_Filter_v1"
timeframe = "4h"
leverage = 1.0