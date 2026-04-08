#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray Momentum with 1w Trend Filter
Hypothesis: Williams Alligator identifies market phases (sleeping/awake/hunting).
Elder Ray measures bull/bear power. Combined, they capture momentum in trending markets.
The 1w trend filter ensures we only trade in strong weekly trends, avoiding whipsaws in ranging markets.
Works in bull/bear by requiring alignment with higher timeframe trend. Targets 15-35 trades/year.
"""

name = "6h_alligator_elder_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for trend filter - call ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 13, 8, 5 SMAs for Williams Alligator (Jaws, Teeth, Lips)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # 8-period SMA
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # 5-period SMA
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 20-period EMA for 1w trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema20_1w[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1w EMA for current 6h bar
        ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)[i]
        
        # Williams Alligator: Mouth open when all lines are separated and ordered
        # Bullish: Lips > Teeth > Jaw (green alignment)
        # Bearish: Jaw > Teeth > Lips (red alignment)
        alligator_bullish = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        alligator_bearish = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Elder Ray: Strong bull power and weak bear power for longs
        # Strong bear power and weak bull power for shorts
        elder_bullish = bull_power[i] > 0 and bear_power[i] < 0
        elder_bearish = bear_power[i] > 0 and bull_power[i] < 0
        
        # 1w trend filter: price above/below 20 EMA
        uptrend_1w = close[i] > ema20_1w_aligned
        downtrend_1w = close[i] < ema20_1w_aligned
        
        if position == 1:  # Long position
            # Exit: alligator closes OR elder ray weakens OR trend fails
            if not (alligator_bullish and elder_bullish and uptrend_1w):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: alligator closes OR elder ray weakens OR trend fails
            if not (alligator_bearish and elder_bearish and downtrend_1w):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter long: alligator aligned bullish + elder bullish + uptrend
            if alligator_bullish and elder_bullish and uptrend_1w:
                position = 1
                signals[i] = 0.25
            # Enter short: alligator aligned bearish + elder bearish + downtrend
            elif alligator_bearish and elder_bearish and downtrend_1w:
                position = -1
                signals[i] = -0.25
    
    return signals