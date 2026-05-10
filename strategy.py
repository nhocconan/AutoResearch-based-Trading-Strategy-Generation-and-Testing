#!/usr/bin/env python3
"""
4h_ChaikinOscillator_Stochastic_Filter
Hypothesis: Chaikin Oscillator crossing zero with Stochastic overbought/oversold levels provides momentum signals.
Chaikin Oscillator measures accumulation/distribution; crosses indicate shifts in buying/selling pressure.
Stochastic filters to avoid trend exhaustion. Works in bull/bear by requiring momentum alignment with price action.
Target: 20-30 trades/year (80-120 total) to minimize fee drag.
"""

name = "4h_ChaikinOscillator_Stochastic_Filter"
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
    
    # Accumulation/Distribution Line
    clv = ((close - low) - (high - close)) / (high - low)
    clv = np.where((high - low) == 0, 0, clv)
    adl = np.cumsum(clv * volume)
    
    # Chaikin Oscillator: (3-day EMA of ADL) - (10-day EMA of ADL)
    def ema(values, period):
        result = np.full_like(values, np.nan, dtype=np.float64)
        if len(values) < period:
            return result
        multiplier = 2 / (period + 1)
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (values[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    adl_ema3 = ema(adl, 3)
    adl_ema10 = ema(adl, 10)
    chaikin = adl_ema3 - adl_ema10
    
    # Stochastic Oscillator (14,3,3)
    def stochastic(high, low, close, k_period=14, d_period=3):
        k = np.full_like(close, np.nan, dtype=np.float64)
        for i in range(k_period-1, len(close)):
            highest_high = np.max(high[i-k_period+1:i+1])
            lowest_low = np.min(low[i-k_period+1:i+1])
            if highest_high - lowest_low == 0:
                k[i] = 50
            else:
                k[i] = (close[i] - lowest_low) / (highest_high - lowest_low) * 100
        
        d = np.full_like(close, np.nan, dtype=np.float64)
        for i in range(len(close)):
            if i < k_period + d_period - 2:
                continue
            start_idx = i - d_period + 1
            if start_idx < k_period-1:
                continue
            valid_k = k[start_idx:i+1]
            if np.all(np.isnan(valid_k)):
                d[i] = np.nan
            else:
                d[i] = np.nanmean(valid_k)
        return k, d
    
    stoch_k, stoch_d = stochastic(high, low, close, 14, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for sufficient data
    
    for i in range(start_idx, n):
        if np.isnan(chaikin[i]) or np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chaikin crossing zero
        chaikin_cross_up = chaikin[i] > 0 and chaikin[i-1] <= 0
        chaikin_cross_down = chaikin[i] < 0 and chaikin[i-1] >= 0
        
        # Stochastic conditions
        stoch_overbought = stoch_k[i] > 80
        stoch_oversold = stoch_k[i] < 20
        
        if position == 0:
            # Long: Chaikin crosses up AND not overbought
            if chaikin_cross_up and not stoch_overbought:
                signals[i] = 0.25
                position = 1
            # Short: Chaikin crosses down AND not oversold
            elif chaikin_cross_down and not stoch_oversold:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Chaikin crosses down OR overbought
            if chaikin_cross_down or stoch_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Chaikin crosses up OR oversold
            if chaikin_cross_up or stoch_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals