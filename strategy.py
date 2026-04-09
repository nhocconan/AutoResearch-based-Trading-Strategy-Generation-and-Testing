#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams %R combo with 1w trend filter
# - Uses 1w EMA(21) for primary trend direction (bull/bear regime)
# - Enters long in bull regime when Elder Bull Power > 0 AND Williams %R < -80 (oversold)
# - Enters short in bear regime when Elder Bear Power < 0 AND Williams %R > -20 (overbought)
# - Uses ATR(14) for dynamic position sizing (volatility-adjusted)
# - Targets 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Elder Ray measures bull/bear power vs EMA13, Williams %R identifies extremes
# - Weekly trend filter ensures we trade with higher timeframe momentum
# - Works in both bull (buy dips) and bear (sell rallies) markets via regime filter

name = "6h_1w_elderray_williamsr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    
    # 1w EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Williams %R(14) on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Align 1d indicators to 6h
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # 6h ATR(14) for volatility normalization
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_6h[0] = tr_6h[0]
    
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(williams_r_1d_aligned[i]) or
            np.isnan(atr_6h[i]) or atr_6h[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1w EMA(21)
        bull_regime = close_6h[i] > ema_21_1w_aligned[i]
        bear_regime = close_6h[i] < ema_21_1w_aligned[i]
        
        # Long conditions: bull regime + bull power positive + Williams oversold
        if bull_regime and bull_power_1d_aligned[i] > 0 and williams_r_1d_aligned[i] < -80:
            signals[i] = 0.25  # 25% position
        
        # Short conditions: bear regime + bear power negative + Williams overbought
        elif bear_regime and bear_power_1d_aligned[i] < 0 and williams_r_1d_aligned[i] > -20:
            signals[i] = -0.25  # 25% short
        
        else:
            signals[i] = 0.0  # Flat
    
    return signals