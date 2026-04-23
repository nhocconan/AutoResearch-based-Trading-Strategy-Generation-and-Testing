#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR volume spike filter and ADX trend regime.
Long when price breaks above upper Donchian channel AND 1d ATR ratio > 1.8 (volatility expansion) AND 1d ADX > 20.
Short when price breaks below lower Donchian channel AND 1d ATR ratio > 1.8 AND 1d ADX > 20.
Exit when price reverts to middle Donchian (20-period average) OR ATR ratio < 1.2 (volatility contraction).
Donchian channels provide structural breakout levels. 1d ATR ratio > 1.8 confirms volatility expansion
typical of genuine breakouts vs false squeezes. 1d ADX > 20 ensures we trade in trending regimes,
avoiding chop. Designed for 12h timeframe targeting 50-150 total trades over 4 years with low frequency
to minimize fee drag. Works in both bull and bear markets by capturing expansion breakouts in trending regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR and ADX filters - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR on 1d data
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d)
    
    # Calculate 30-period average ATR for ratio
    atr_ma_30 = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_1d / (atr_ma_30 + 1e-10)  # Avoid division by zero
    
    # Calculate ADX on 1d data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr_tr = np.zeros_like(tr)
        atr_tr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr_tr[i] = (atr_tr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr_tr[i] != 0:
                plus_di[i] = (np.sum(plus_dm[i-period+1:i+1]) / atr_tr[i]) * 100
                minus_di[i] = (np.sum(minus_dm[i-period+1:i+1]) / atr_tr[i]) * 100
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d)
    
    # Align 1d indicators to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channels on 12h timeframe
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    donch_upper, donch_lower, donch_middle = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(donch_middle[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio_val = atr_ratio_aligned[i]
        adx_val = adx_1d_aligned[i]
        upper = donch_upper[i]
        lower = donch_lower[i]
        middle = donch_middle[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND volatility expansion (ATR ratio > 1.8) AND trending (ADX > 20)
            if (price > upper and atr_ratio_val > 1.8 and adx_val > 20):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian AND volatility expansion AND trending
            elif (price < lower and atr_ratio_val > 1.8 and adx_val > 20):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle Donchian OR volatility contraction (ATR ratio < 1.2)
                if (price < middle or atr_ratio_val < 1.2):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle Donchian OR volatility contraction
                if (price > middle or atr_ratio_val < 1.2):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dATRratio_ADX_Volume"
timeframe = "12h"
leverage = 1.0