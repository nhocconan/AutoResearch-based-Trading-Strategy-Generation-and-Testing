#!/usr/bin/env python3
"""
exp_6450_1d_donchian20_1w_hma_vol_v1
Hypothesis: Daily Donchian(20) breakout with 1-week HMA trend filter and volume confirmation.
Works in bull/bear: HMA filters false breakouts in ranging markets, volume confirms institutional interest.
Target: 15-25 trades/year via tight Donchian breakout + volume spike requirement.
"""
name = "exp_6450_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Daily Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 2. Load 1-week HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # 3. Calculate 1-week HMA(21)
    close_1w = df_1w['close'].values
    hma_21 = calculate_hma(close_1w, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # 4. Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if volume not confirmed
        if volume[i] < 1.5 * vol_ma[i]:
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above Donchian high + 1w HMA uptrend
        if close[i] > donchian_high[i] and hma_21_aligned[i] > hma_21_aligned[i-1]:
            if position != 1:
                signals[i] = 0.30  # Enter long 30%
                position = 1
                entry_price = close[i]
            else:
                signals[i] = 0.30  # Hold long
        # Short conditions: price breaks below Donchian low + 1w HMA downtrend
        elif close[i] < donchian_low[i] and hma_21_aligned[i] < hma_21_aligned[i-1]:
            if position != -1:
                signals[i] = -0.30  # Enter short 30%
                position = -1
                entry_price = close[i]
            else:
                signals[i] = -0.30  # Hold short
        else:
            # Exit conditions: reverse signal or stoploss
            if position == 1:
                # Stoploss: 2*ATR below entry
                atr_val = calculate_atr(high, low, close, 14, i)
                if atr_val > 0 and close[i] < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                # Reverse signal
                elif close[i] < donchian_low[i] and hma_21_aligned[i] < hma_21_aligned[i-1]:
                    signals[i] = -0.30
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.30
            elif position == -1:
                # Stoploss: 2*ATR above entry
                atr_val = calculate_atr(high, low, close, 14, i)
                if atr_val > 0 and close[i] > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                # Reverse signal
                elif close[i] > donchian_high[i] and hma_21_aligned[i] > hma_21_aligned[i-1]:
                    signals[i] = 0.30
                    position = 1
                    entry_price = close[i]
                else:
                    signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

def calculate_hma(array, period):
    """Hull Moving Average"""
    if len(array) < period:
        return np.full_like(array, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(array).rolling(window=half, min_periods=half).mean().values
    wma1 = pd.Series(array).rolling(window=period, min_periods=period).mean().values
    raw_hma = 2 * wma2 - wma1
    hma = pd.Series(raw_hma).rolling(window=sqrt, min_periods=sqrt).mean().values
    return hma

def calculate_atr(high, low, close, period, current_idx):
    """ATR calculation for stoploss"""
    if current_idx < period:
        return 0.0
    tr1 = high[current_idx:] - low[current_idx:]
    tr2 = np.abs(high[current_idx:] - np.append([close[0]], close[current_idx:-1]))
    tr3 = np.abs(low[current_idx:] - np.append([close[0]], close[current_idx:-1]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    if len(tr) < period:
        return np.mean(tr)
    return np.mean(tr[-period:])