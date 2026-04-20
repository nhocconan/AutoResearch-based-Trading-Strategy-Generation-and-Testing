#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with Weekly Volatility Filter
# - Long when price breaks above 20-period Donchian high + weekly ATR < monthly ATR (low volatility regime)
# - Short when price breaks below 20-period Donchian low + weekly ATR < monthly ATR
# - Uses 1d and 1w timeframes for trend and volatility context
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d and 1w data for context
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels on 12h
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR on 1d and 1w for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1_w = np.abs(high_1w - low_1w)
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr1_w[0] = tr2_w[0] = tr3_w[0] = 0
    tr_1w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate monthly ATR (4-week) for volatility comparison
    atr_4w = pd.Series(tr_1w).rolling(window=56, min_periods=56).mean().values  # ~4 weeks
    
    # Align indicators to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, pd.DataFrame({'high': high_12h, 'low': low_12h}), donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, pd.DataFrame({'high': high_12h, 'low': low_12h}), donchian_low)
    atr_1d_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_1w_12h = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_4w_12h = align_htf_to_ltf(prices, df_1w, atr_4w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or \
           np.isnan(atr_1d_12h[i]) or np.isnan(atr_1w_12h[i]) or np.isnan(atr_4w_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime: weekly ATR < monthly ATR (low volatility environment)
        low_vol_regime = atr_1w_12h[i] < atr_4w_12h[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + low volatility regime
            if close_12h[i] > donchian_high_12h[i] and low_vol_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + low volatility regime
            elif close_12h[i] < donchian_low_12h[i] and low_vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or volatility expands
            if close_12h[i] < donchian_low_12h[i] or not low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or volatility expands
            if close_12h[i] > donchian_high_12h[i] or not low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolatilityFilter"
timeframe = "12h"
leverage = 1.0