#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_MA_Crossover
Hypothesis: Combines Elder Ray's Bull/Bear Power with a zero-lag moving average crossover on 6h timeframe, filtered by 1d trend direction. Bull Power = Close - EMA(13), Bear Power = EMA(13) - High. Uses zero-lag MACD for timely signals with reduced lag. Designed for 6h timeframe to capture medium-term swings in both bull and bear markets with moderate trade frequency (~25-35 trades/year).
"""

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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate EMA13 for Elder Ray and zero-lag MA
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power and Bear Power
    bull_power = close - ema13
    bear_power = ema13 - high
    
    # Zero-lag EMA: EMA + (EMA - EMA lagged)
    ema13_lag = np.roll(ema13, 1)
    ema13_lag[0] = ema13[0]
    zl_ema = ema13 + (ema13 - ema13_lag)
    
    # Signal line for zero-lag MACD (EMA of zl_ema)
    signal_line = pd.Series(zl_ema).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for indicators
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(zl_ema[i]) or np.isnan(signal_line[i])):
            signals[i] = 0.0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        zl = zl_ema[i]
        signal = signal_line[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        if position == 0:
            # Long: zero-lag MA crosses above signal, Bull Power > 0, above 1d EMA34
            if zl > signal and bull > 0 and close[i] > ema34_val:
                signals[i] = size
                position = 1
            # Short: zero-lag MA crosses below signal, Bear Power > 0, below 1d EMA34
            elif zl < signal and bear > 0 and close[i] < ema34_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: zero-lag MA crosses below signal OR Bear Power > 0
            if zl < signal or bear > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: zero-lag MA crosses above signal OR Bull Power > 0
            if zl > signal or bull > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_ZeroLag_MA_Crossover"
timeframe = "6h"
leverage = 1.0