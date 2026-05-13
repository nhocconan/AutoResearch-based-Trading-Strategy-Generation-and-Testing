#!/usr/bin/env python3
name = "12H_AdaptiveKeltner_Donchian_1dTrend_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR (14-period) for Keltner channels
    atr = np.zeros(n)
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[0], tr])
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate EMA (20-period) for Keltner center
    ema_20 = np.zeros(n)
    ema_20[:] = np.nan
    alpha = 2 / (20 + 1)
    for i in range(len(close)):
        if i == 0:
            ema_20[i] = close[i]
        elif np.isnan(ema_20[i-1]):
            ema_20[i] = close[i]
        else:
            ema_20[i] = alpha * close[i] + (1 - alpha) * ema_20[i-1]
    
    # Keltner Channel: upper/lower bands (ATR multiplier = 2.0)
    keltner_upper = ema_20 + 2.0 * atr
    keltner_lower = ema_20 - 2.0 * atr
    
    # Donchian Channel (20-period)
    donchian_upper = np.zeros(n)
    donchian_lower = np.zeros(n)
    donchian_upper[:] = np.nan
    donchian_lower[:] = np.nan
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20:i+1])
        donchian_lower[i] = np.min(low[i-20:i+1])
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA50 on daily close
    ema_50 = np.zeros_like(close_1d)
    ema_50[:] = np.nan
    alpha_50 = 2 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_50[i] = close_1d[i]
        elif np.isnan(ema_50[i-1]):
            ema_50[i] = close_1d[i]
        else:
            ema_50[i] = alpha_50 * close_1d[i] + (1 - alpha_50) * ema_50[i-1]
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Adaptive condition: Price must be outside both Keltner and Donchian channels
        outside_keltner = close[i] > keltner_upper[i] or close[i] < keltner_lower[i]
        outside_donchian = close[i] > donchian_upper[i] or close[i] < donchian_lower[i]
        
        if position == 0:
            # LONG: Price breaks above both channels + 1d uptrend
            if close[i] > keltner_upper[i] and close[i] > donchian_upper[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below both channels + 1d downtrend
            elif close[i] < keltner_lower[i] and close[i] < donchian_lower[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Keltner channel (mean reversion signal)
            if keltner_lower[i] < close[i] < keltner_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Keltner channel
            if keltner_lower[i] < close[i] < keltner_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals