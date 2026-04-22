#!/usr/bin/env python3
"""
Hypothesis: 4-hour KAMA (Kaufman Adaptive Moving Average) with 12-hour trend filter and volume spike.
Long when KAMA > price and 12h EMA50 rising with volume spike.
Short when KAMA < price and 12h EMA50 falling with volume spike.
Exit when price crosses KAMA or 12h EMA50 reverses.
KAMA adapts to market noise, reducing whipsaws in choppy markets. Combined with 12h trend and volume,
it filters false signals and captures strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_length=10, fast=2, slow=30):
        # Calculate Efficiency Ratio
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        # Handle first er_length elements
        er = np.full_like(close, np.nan, dtype=float)
        for i in range(er_length, len(close)):
            if volatility[i] != 0:
                er[i] = change[i-er_length] / volatility[i-er_length+1:i+1].sum()
            else:
                er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Calculate KAMA
        kama_vals = np.full_like(close, np.nan, dtype=float)
        kama_vals[er_length] = close[er_length]
        for i in range(er_length+1, len(close)):
            if not np.isnan(sc[i]):
                kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
            else:
                kama_vals[i] = kama_vals[i-1]
        return kama_vals
    
    kama_vals = kama(close, er_length=10, fast=2, slow=30)
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price > KAMA and 12h EMA50 rising with volume spike
            if (close[i] > kama_vals[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA and 12h EMA50 falling with volume spike
            elif (close[i] < kama_vals[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses KAMA or 12h EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= KAMA or 12h EMA50 turns down
                if close[i] <= kama_vals[i] or ema50_12h_aligned[i] < ema50_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price >= KAMA or 12h EMA50 turns up
                if close[i] >= kama_vals[i] or ema50_12h_aligned[i] > ema50_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_KAMA_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0