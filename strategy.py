#!/usr/bin/env python3
name = "1d_Williams_Alligator_ElderRay_Trend"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Williams Alligator: 3 SMAs (Jaw: 13, Teeth: 8, Lips: 5)
    # Calculate on weekly close
    close_1w = df_1w['close'].values
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().values
    
    # Align to daily timeframe with proper delay
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (on daily)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for Alligator to form
    
    for i in range(start_idx, n):
        # Skip if any Alligator line is not available
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator aligned (Lips > Teeth > Jaw) AND Bull Power > 0
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and bull_power[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned (Lips < Teeth < Jaw) AND Bear Power < 0
            elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and bear_power[i] < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator starts to reverse (Lips < Teeth) OR Bull Power <= 0
            if lips_aligned[i] < teeth_aligned[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator starts to reverse (Lips > Teeth) OR Bear Power >= 0
            if lips_aligned[i] > teeth_aligned[i] or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Alligator identifies trending vs ranging markets on weekly timeframe.
# Elder Ray confirms daily trend strength via bull/bear power relative to EMA13.
# Long when weekly Alligator is bullish (Lips>Teeth>Jaw) AND daily Bull Power > 0.
# Short when weekly Alligator is bearish (Lips<Teeth<Jaw) AND daily Bear Power < 0.
# Weekly timeframe reduces noise and avoids whipsaws, daily provides timely signals.
# Discrete 0.25 position size limits risk in volatile markets.
# Works in bull markets (Alligator alignment + positive power) and bear markets (reverse).
# Target: 15-25 trades/year to minimize fee decay while capturing major trends.