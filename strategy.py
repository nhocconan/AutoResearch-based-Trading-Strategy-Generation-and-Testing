#!/usr/bin/env python3
name = "1d_Williams_Alligator_ElderRay_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w Williams Alligator: Jaw (13), Teeth (8), Lips (5)
    jaw = pd.Series(df_1w['close']).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(df_1w['close']).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(df_1w['close']).rolling(window=5, min_periods=5).mean().values
    jaw_aligned = align_ltf_to_htf(prices, df_1w, jaw)
    teeth_aligned = align_ltf_to_htf(prices, df_1w, teeth)
    lips_aligned = align_ltf_to_htf(prices, df_1w, lips)
    
    # 1w Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(df_1w['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1w['high'].values - ema13
    bear_power = df_1w['low'].values - ema13
    bull_power_aligned = align_ltf_to_htf(prices, df_1w, bull_power)
    bear_power_aligned = align_ltf_to_htf(prices, df_1w, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for Alligator
    
    for i in range(start_idx, n):
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or \
           np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0
            if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and bull_power_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) AND Bear Power < 0
            elif jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and bear_power_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Lips < Jaw (Alligator sleeping) OR Bull Power < 0
            if lips_aligned[i] < jaw_aligned[i] or bull_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Jaws < Lips (Alligator sleeping) OR Bear Power > 0
            if jaw_aligned[i] < lips_aligned[i] or bear_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Alligator identifies trend presence and direction via SMAs (13,8,5).
# Elder Ray measures bull/bear power relative to EMA13. Together they filter whipsaws.
# Long when Alligator is bullish (Lips>Teeth>Jaw) and Bull Power > 0.
# Short when Alligator is bearish (Jaw>Teeth>Lips) and Bear Power < 0.
# Weekly timeframe reduces noise, daily execution captures trends.
# Discrete 0.25 position size limits drawdown in choppy markets.
# Works in bull markets (trend following) and bear markets (reverse criteria).
# Target: 15-30 trades/year to minimize fee drag while capturing sustained moves.