#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ehlers Fisher Transform with 12-hour trend filter and volume confirmation.
Long when Fisher crosses above -1.5 during 12-hour uptrend with volume spike.
Short when Fisher crosses below +1.5 during 12-hour downtrend with volume spike.
Exit when Fisher crosses back through zero or trend reverses.
Designed for low-to-moderate trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 12-hour trend.
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
    
    # Ehlers Fisher Transform on 6h prices (length=10)
    def ehlers_fisher_transform(price_series, length=10):
        n = len(price_series)
        if n < length:
            return np.full(n, np.nan), np.full(n, np.nan)
        
        # Normalize price to [-1, 1] range over lookback period
        highest = np.maximum.accumulate(price_series)
        lowest = np.minimum.accumulate(price_series)
        range_val = highest - lowest
        range_val = np.where(range_val == 0, 1, range_val)  # avoid division by zero
        
        value1 = 2 * ((price_series - lowest) / range_val - 0.5)
        value1 = np.clip(value1, -0.999, 0.999)  # avoid log(0)
        
        # Smooth with exponential moving average
        alpha = 2.0 / (length + 1)
        value2 = np.zeros(n)
        value2[0] = value1[0]
        for i in range(1, n):
            value2[i] = alpha * value1[i] + (1 - alpha) * value2[i-1]
        
        # Fisher transform
        fish = 0.5 * np.log((1 + value2) / (1 - value2))
        fish = np.where(np.isnan(value2) | np.isinf(value2), np.nan, fish)
        
        # Signal line (3-period EMA of Fisher)
        signal = np.zeros(n)
        signal[0] = fish[0] if not np.isnan(fish[0]) else 0
        for i in range(1, n):
            if np.isnan(fish[i]):
                signal[i] = signal[i-1]
            else:
                signal[i] = alpha * fish[i] + (1 - alpha) * signal[i-1]
        
        return fish, signal
    
    fish, fish_signal = ehlers_fisher_transform(close, length=10)
    
    # Load 12-hour data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 20-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(fish[i]) or np.isnan(fish_signal[i]) or 
            np.isnan(ema20_12h_aligned[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_30[i]
        
        if position == 0:
            # Long: Fisher crosses above -1.5 + 12h uptrend + volume spike
            if fish[i] > -1.5 and fish_signal[i] <= -1.5 and ema20_12h_aligned[i] > ema20_12h_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below +1.5 + 12h downtrend + volume spike
            elif fish[i] < 1.5 and fish_signal[i] >= 1.5 and ema20_12h_aligned[i] < ema20_12h_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Fisher crosses back through zero or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Fisher crosses below zero or 12h trend turns down
                if fish[i] < 0 and fish_signal[i] >= 0 or ema20_12h_aligned[i] < ema20_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Fisher crosses above zero or 12h trend turns up
                if fish[i] > 0 and fish_signal[i] <= 0 or ema20_12h_aligned[i] > ema20_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ehlers_Fisher_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0