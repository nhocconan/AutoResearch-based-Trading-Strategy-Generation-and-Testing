#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
# Long when price breaks above Donchian upper band with ATR(30)/ATR(7) < 0.8 (low volatility regime) and volume > 1.5x average.
# Short when price breaks below Donchian lower band with ATR(30)/ATR(7) < 0.8 and volume > 1.5x average.
# Uses discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# Donchian channels provide clear breakout levels. ATR filter ensures we trade in low volatility regimes before expansion.
# Volume spike confirms participation. Works in trending markets via breakouts and avoids choppy conditions.

name = "4h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "4h"
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
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    if n < lookback + 1:
        return np.zeros(n)
    
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR (14-period) on 1d data
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    atr_7_1d = calculate_atr(high_1d, low_1d, close_1d, 7)
    atr_30_1d = calculate_atr(high_1d, low_1d, close_1d, 30)
    
    # Avoid division by zero
    atr_ratio = np.where(atr_7_1d > 0, atr_30_1d / atr_7_1d, 1.0)
    
    # Align 1d ATR ratio to 4h timeframe (wait for 1d bar to close)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper band with low volatility regime (ATR30/ATR7 < 0.8) and volume spike
            if (close[i] > upper[i] and 
                atr_ratio_aligned[i] < 0.8 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower band with low volatility regime and volume spike
            elif (close[i] < lower[i] and 
                  atr_ratio_aligned[i] < 0.8 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower band (reversal signal) OR volatility expands (ATR ratio > 1.2)
            if (close[i] < lower[i]) or (atr_ratio_aligned[i] > 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper band (reversal signal) OR volatility expands (ATR ratio > 1.2)
            if (close[i] > upper[i]) or (atr_ratio_aligned[i] > 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals