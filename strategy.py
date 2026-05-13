#!/usr/bin/env python3
"""
6h_Elder_Ray_Power_Reversal
Hypothesis: Elder Ray Index (Bull/Bear Power) with 13-period EMA captures institutional buying/selling pressure.
In bull markets: buy when Bear Power turns positive after negative (selling exhaustion).
In bear markets: sell when Bull Power turns negative after positive (buying exhaustion).
Uses 1-week trend filter to avoid counter-trend trades. Target: 15-30 trades/year per symbol.
"""

name = "6h_Elder_Ray_Power_Reversal"
timeframe = "6h"
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
    
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13   # Bull Power = High - EMA13
    bear_power = low - ema_13    # Bear Power = Low - EMA13
    
    # 6h trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_6h = close > ema_50
    downtrend_6h = close < ema_50
    
    # 1-week trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        bull = bull_power[i]
        bear = bear_power[i]
        uptrend = uptrend_6h[i]
        downtrend = downtrend_6h[i]
        uptrend_htf = uptrend_1w_aligned[i]
        downtrend_htf = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: Bear Power turns positive after being negative (selling exhaustion)
            # Only in 1-week uptrend
            if bear > 0 and bear_power[i-1] <= 0 and uptrend and uptrend_htf:
                signals[i] = 0.25
                position = 1
            # SHORT: Bull Power turns negative after being positive (buying exhaustion)
            # Only in 1-week downtrend
            elif bull < 0 and bull_power[i-1] >= 0 and downtrend and downtrend_htf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative or 6h trend turns down
            if bull < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive or 6h trend turns up
            if bear > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals