#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + Elder Ray Force Index with 1d volume confirmation
# Long when green line > red line (bullish alignment) and Force Index > 0 with volume spike
# Short when red line > green line (bearish alignment) and Force Index < 0 with volume spike
# Exit when Alligator lines converge (market sleeping) or Force Index crosses zero
# Uses Alligator for trend direction, Elder Ray for momentum, volume for confirmation
# Designed to capture sustained trends while avoiding whipsaws in ranging markets
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "12h_Alligator_ElderRay_PowerTrend"
timeframe = "12h"
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
    
    # Calculate 1d Williams Alligator (Jaw, Teeth, Lips)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Alligator lines: SMAs of median price
    median_price = (df_1d['high'] + df_1d['low']) / 2
    jaw = median_price.rolling(window=13, center=False, min_periods=13).mean().shift(8)
    teeth = median_price.rolling(window=8, center=False, min_periods=8).mean().shift(5)
    lips = median_price.rolling(window=5, center=False, min_periods=5).mean().shift(3)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    
    # Calculate 1d Elder Ray Force Index (Close - Prior Close) * Volume
    price_change = df_1d['close'].diff()
    force_index = (price_change * df_1d['volume']).values
    force_index_aligned = align_htf_to_ltf(prices, df_1d, force_index)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Alligator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(force_index_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish alignment (lips > teeth > jaw) and positive Force Index with volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                force_index_aligned[i] > 0 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment (jaw > teeth > lips) and negative Force Index with volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and 
                  force_index_aligned[i] < 0 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator sleeping (lines converge) or Force Index turns negative
            lips_teeth_diff = abs(lips_aligned[i] - teeth_aligned[i])
            teeth_jaw_diff = abs(teeth_aligned[i] - jaw_aligned[i])
            convergence = (lips_teeth_diff < 0.001 * close[i]) or (teeth_jaw_diff < 0.001 * close[i])
            if convergence or force_index_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator sleeping (lines converge) or Force Index turns positive
            lips_teeth_diff = abs(lips_aligned[i] - teeth_aligned[i])
            teeth_jaw_diff = abs(teeth_aligned[i] - jaw_aligned[i])
            convergence = (lips_teeth_diff < 0.001 * close[i]) or (teeth_jaw_diff < 0.001 * close[i])
            if convergence or force_index_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals