#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volatility regime filter.
- Long: Close > Donchian Upper(20) AND price > 1d EMA50 AND ATR(14) < ATR(50) (low volatility breakout)
- Short: Close < Donchian Lower(20) AND price < 1d EMA50 AND ATR(14) < ATR(50)
- Exit: Opposite Donchian breakout OR price crosses 1d EMA50
- Uses 1d HTF for EMA50 trend filter
- Designed for low trade frequency (19-50/year) to minimize fee drag on 4h timeframe
- ATR regime filter ensures breakouts occur during low volatility compression (bull/bear agnostic)
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
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility regime filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 50)  # Need 50 for EMA/ATR, 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(atr_50[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: ATR(14) < ATR(50) (low volatility environment)
        low_vol_regime = atr_14[i] < atr_50[i]
        
        # Donchian breakout signals (using current close vs prior channels)
        breakout_up = close[i] > highest_20[i-1]  # Close above prior upper channel
        breakout_down = close[i] < lowest_20[i-1]  # Close below prior lower channel
        
        if position == 0:
            # Long: Donchian upper breakout AND price > 1d EMA50 AND low volatility regime
            if breakout_up and low_vol_regime and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian lower breakout AND price < 1d EMA50 AND low volatility regime
            elif breakout_down and low_vol_regime and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian lower breakout OR price < 1d EMA50 (trend flip)
            if breakout_down or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian upper breakout OR price > 1d EMA50 (trend flip)
            if breakout_up or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_ATRVolRegime"
timeframe = "4h"
leverage = 1.0