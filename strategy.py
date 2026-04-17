#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d ATR for volatility-based stoploss ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 14:
            atr_1d[i] = np.mean(tr[i-14:i+1])
        elif i > 0:
            atr_1d[i] = np.mean(tr[1:i+1])
        else:
            atr_1d[i] = np.nan
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d EMA50 for trend filter ===
    ema50_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i < 50:
            ema50_1d[i] = np.nan
        else:
            if i == 50:
                ema50_1d[i] = np.mean(close_1d[:51])
            else:
                ema50_1d[i] = (close_1d[i] * 2 / 51) + ema50_1d[i-1] * (49 / 51)
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h Donchian(20) channels ===
    donchian_high = np.full_like(close, np.nan)
    donchian_low = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i+1])
            donchian_low[i] = np.min(low[i-20:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high in uptrend (price > EMA50)
            if close[i] > donchian_high[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low in downtrend (price < EMA50)
            elif close[i] < donchian_low[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price falls below Donchian low OR stoploss hit
            if close[i] < donchian_low[i] or close[i] < (ema50_1d_aligned[i] - 2.0 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above Donchian high OR stoploss hit
            if close[i] > donchian_high[i] or close[i] > (ema50_1d_aligned[i] + 2.0 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_EMA50_VolumeFilter"
timeframe = "12h"
leverage = 1.0