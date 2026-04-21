# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_WilliamsAlligator_Trend_With_Volume
Hypothesis: Use Williams Alligator (3 SMAs) on 1d timeframe to filter trend direction,
enter on Alligator alignment + price outside lips + volume confirmation on 12h timeframe.
Exit when price re-enters Alligator mouth or trend changes.
Williams Alligator catches strong trends while avoiding choppy markets.
Works in bull markets by riding uptrends and in bear markets by riding downtrends.
Volume confirmation ensures institutional participation. Target: 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # === Williams Alligator on daily timeframe ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Median price (typical price)
    median_price_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 12h Volume confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: Alligator aligned upward (Lips > Teeth > Jaw) + price above Lips + volume
            if (lips_val > teeth_val and 
                teeth_val > jaw_val and 
                price_close > lips_val and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned downward (Lips < Teeth < Jaw) + price below Lips + volume
            elif (lips_val < teeth_val and 
                  teeth_val < jaw_val and 
                  price_close < lips_val and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price re-enters Alligator mouth (between Lips and Jaw) or trend changes
            if position == 1:
                # Exit long if price falls below Lips or Alligator alignment breaks
                if price_close < lips_val or not (lips_val > teeth_val and teeth_val > jaw_val):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if price rises above Lips or Alligator alignment breaks
                if price_close > lips_val or not (lips_val < teeth_val and teeth_val < jaw_val):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Trend_With_Volume"
timeframe = "12h"
leverage = 1.0