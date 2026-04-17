#!/usr/bin/env python3
"""
12h_1d_Williams_Alligator_ElderRay_V1
Hypothesis: On 12h timeframe, use Williams Alligator (jaw/teeth/lips) for trend direction and Elder Ray (bull/bear power) for momentum confirmation. Enter long when price > teeth and bull power > 0 with expanding teeth (bullish alignment), short when price < teeth and bear power < 0 with contracting teeth (bearish alignment). Use 1d ATR for volatility filtering and position sizing. Designed for low frequency (target 15-30 trades/year) to work in both bull and bear markets by capturing sustained trends with momentum confirmation.
"""

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
    
    # === 1d Data (HTF for ATR and Elder Ray) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Williams Alligator (12h)
    def calculate_ema(arr, period):
        ema = np.full_like(arr, np.nan)
        if len(arr) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema[i] = (arr[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    # Williams Alligator on 12h
    jaw = calculate_ema(close, 13)  # 13-period
    teeth = calculate_ema(close, 8)  # 8-period
    lips = calculate_ema(close, 5)   # 5-period
    
    # Elder Ray on 1d (13-period EMA)
    ema_13_1d = calculate_ema(close_1d, 13)
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align 1d data to 12h
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 1d ATR for volatility filter and position sizing
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) < period:
            return atr
        atr[period-1] = np.nanmean(tr[1:period])
        for i in range(period, len(tr)):
            if not np.isnan(tr[i]):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Williams Alligator conditions
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Elder Ray conditions
        bullish_momentum = bull_power_aligned[i] > 0
        bearish_momentum = bear_power_aligned[i] < 0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish alignment + bullish momentum
            if bullish_alignment and bullish_momentum and close[i] > teeth[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish alignment + bearish momentum
            elif bearish_alignment and bearish_momentum and close[i] < teeth[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit: bearish alignment OR loss of bullish momentum
            if bearish_alignment or not bullish_momentum:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: bullish alignment OR loss of bearish momentum
            if bullish_alignment or not bearish_momentum:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Williams_Alligator_ElderRay_V1"
timeframe = "12h"
leverage = 1.0