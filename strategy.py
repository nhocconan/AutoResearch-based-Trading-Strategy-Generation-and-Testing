#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based position sizing.
- Long: Close > Donchian(20) high AND price > 1d EMA50
- Short: Close < Donchian(20) low AND price < 1d EMA50
- Exit: Opposite Donchian breakout OR price crosses 1d EMA50
- Position size: 0.25 when ATR(14) < median ATR(50), else 0.15 (reduce size in high vol)
- Uses 1d HTF for EMA50 (calculated from prior completed 1d bar)
- Designed for low trade frequency (~20-50/year) to minimize fee drag
- Works in bull (buy breakouts above Donchian high) and bear (sell breakdowns below Donchian low)
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
    
    # Donchian(20) - 20-period high/low
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility-based position sizing
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_median = pd.Series(atr).rolling(window=50, min_periods=50).median().values
    
    # Calculate 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for Donchian, 14 for ATR, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_median[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Dynamic position size based on volatility
        if atr[i] < atr_median[i]:
            size = 0.25  # Normal size in low volatility
        else:
            size = 0.15  # Reduced size in high volatility
        
        # Donchian breakout signals (using current close vs prior levels)
        breakout_up = close[i] > high_roll[i-1]   # Close above prior Donchian high
        breakout_down = close[i] < low_roll[i-1]  # Close below prior Donchian low
        
        if position == 0:
            # Long: Donchian breakout up AND price > 1d EMA50
            if breakout_up and close[i] > ema_50_1d_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Donchian breakout down AND price < 1d EMA50
            elif breakout_down and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long exit: Donchian breakout down OR price < 1d EMA50 (trend flip)
            if breakout_down or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Donchian breakout up OR price > 1d EMA50 (trend flip)
            if breakout_up or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_ATRSize"
timeframe = "4h"
leverage = 1.0