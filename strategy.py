#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Alligator_ElderRay_Ratio"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Alligator: SMAs with periods 13, 8, 5 and shifts 8, 5, 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_price_1d = (high_1d + low_1d) / 2
    
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d := df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Alligator lines and Elder Ray to 6t
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 6h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator aligned (jaws < teeth < lips) + Bull Power > 0 + price above EMA50
            long_cond = (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and 
                         bull_power_aligned[i] > 0 and close[i] > ema50[i])
            
            # Short: Alligator inverted (jaws > teeth > lips) + Bear Power < 0 + price below EMA50
            short_cond = (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and 
                          bear_power_aligned[i] < 0 and close[i] < ema50[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator death cross (jaws > lips) or Bear Power > 0
            if jaw_aligned[i] > lips_aligned[i] or bear_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator death cross (jaws < lips) or Bull Power < 0
            if jaw_aligned[i] < lips_aligned[i] or bull_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Alligator identifies trend alignment (jaws-teeth-lips order) while Elder Ray measures
# bull/bear power relative to EMA13. Long when Alligator is bullish (jaws<teeth<lips) with positive
# Bull Power and price above EMA50. Short when Alligator is bearish (jaws>teeth>lips) with negative
# Bear Power and price below EMA50. Uses 1d indicators aligned to 6t to avoid look-ahead. 
# Works in trending markets (Alligator alignment) and avoids chop (requires EMA50 filter).
# Target: 20-40 trades/year to stay within frequency limits while capturing strong trends.