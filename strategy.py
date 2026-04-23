#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR volatility filter and 1w EMA50 trend filter.
- Long: Close > Donchian High(20) AND 1d ATR < 1w ATR * 0.8 (low volatility regime) AND price > 1w EMA50 (uptrend)
- Short: Close < Donchian Low(20) AND 1d ATR < 1w ATR * 0.8 (low volatility regime) AND price < 1w EMA50 (downtrend)
- Exit: Opposite Donchian breakout OR volatility expansion (1d ATR > 1w ATR * 1.2)
- Uses 1d ATR as volatility filter to avoid false breakouts in high volatility
- 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Works in bull markets (trend continuation) and bear markets (avoids whipsaws via volatility filter)
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
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # ATR calculation (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR (14-period)
    tr1_1d = np.abs(high_1d - low_1d)
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w ATR for volatility regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1_1w = np.abs(high_1w - low_1w)
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    tr_1w[0] = tr1_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for Donchian, 14 for ATR, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr_14[i]) or 
            np.isnan(atr_1d[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime: low volatility (1d ATR < 1w ATR * 0.8) for breakout validity
        low_volatility = atr_1d[i] < atr_1w_aligned[i] * 0.8
        # Volatility expansion filter for exit (1d ATR > 1w ATR * 1.2)
        volatility_expansion = atr_1d[i] > atr_1w_aligned[i] * 1.2
        
        if position == 0:
            # Long: Donchian breakout up + low volatility + uptrend
            if (close[i] > donchian_high[i] and 
                low_volatility and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + low volatility + downtrend
            elif (close[i] < donchian_low[i] and 
                  low_volatility and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown OR volatility expansion
            if close[i] < donchian_low[i] or volatility_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout up OR volatility expansion
            if close[i] > donchian_high[i] or volatility_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dATR_Filter_1wEMA50_Trend"
timeframe = "6h"
leverage = 1.0