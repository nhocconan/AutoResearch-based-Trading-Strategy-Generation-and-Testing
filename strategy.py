#!/usr/bin/env python3
"""
6h_Williams_Alligator_Trend_Filter_with_Volume
Hypothesis: Use Williams Alligator (3 SMAs) from 12h to determine trend direction, enter on 6s breakout of Alligator teeth with volume confirmation. Exit when price re-enters the Alligator's mouth. Designed for 6h to capture medium-term trends with low trade frequency. Works in bull (trend follow) and bear (avoid false signals via Alligator alignment).
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
    
    # Williams Alligator from 12h: Jaw (13), Teeth (8), Lips (5) - all SMAs
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    # Align to 6h timeframe (wait for 12h close)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need Alligator components and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: jaws < teeth < lips = downtrend, jaws > teeth > lips = uptrend
        is_uptrend = jaw_val > teeth_val and teeth_val > lips_val
        is_downtrend = jaw_val < teeth_val and teeth_val < lips_val
        
        if position == 0:
            # Long: price > lips in uptrend with volume spike
            if is_uptrend and price > lips_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < jaws in downtrend with volume spike
            elif is_downtrend and price < jaw_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < teeth (re-enter Alligator's mouth) or trend change
            if price < teeth_val or not is_uptrend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > teeth (re-enter Alligator's mouth) or trend change
            if price > teeth_val or not is_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Williams_Alligator_Trend_Filter_with_Volume"
timeframe = "6h"
leverage = 1.0