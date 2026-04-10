#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator (jaw=13, teeth=8, lips=5) from 1d to define trend direction and market state
# - Alligator sleeping (jaw/teeth/lips intertwined) = ranging market → fade extremes at Camarilla H3/L3
# - Alligator awakening (jaws diverging) = trending market → breakout in direction of jaw alignment
# - Volume confirmation: current 12h volume > 1.8x 30-period average to avoid false breakouts
# - Designed for 12h timeframe: targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: Alligator adapts to trending/ranging regimes
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "12h_1d_alligator_camarilla_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Williams Alligator
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_price_1d = (high_1d + low_1d) / 2.0
    
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price_1d, 13)
    teeth = smma(median_price_1d, 8)
    lips = smma(median_price_1d, 5)
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Pre-compute 1d Camarilla levels (H3/L3 for mean reversion, H4/L4 for breakout)
    rango_1d = high_1d - low_1d
    camarilla_h3 = close_1d + (rango_1d * 1.1 / 4)
    camarilla_l3 = close_1d - (rango_1d * 1.1 / 4)
    camarilla_h4 = close_1d + (rango_1d * 1.1 / 2)
    camarilla_l4 = close_1d - (rango_1d * 1.1 / 2)
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_30 = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume_12h > (1.8 * avg_volume_30)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Alligator state: sleeping (intertwined) vs awakening (diverging)
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Alligator sleeping: all lines close together (market ranging)
        max_line = max(jaw_val, teeth_val, lips_val)
        min_line = min(jaw_val, teeth_val, lips_val)
        alligator_sleeping = (max_line - min_line) < (0.001 * close_1d[i])  # 0.1% threshold
        
        # Alligator awakening: jaws aligned with trend
        jaw_above_teeth = jaw_val > teeth_val
        teeth_above_lips = teeth_val > lips_val
        jaw_below_teeth = jaw_val < teeth_val
        teeth_below_lips = teeth_val < lips_val
        
        if position == 1:  # Long position
            # Exit: price closes below L3 (mean reversion failure) or Alligator reverses
            if prices['close'].iloc[i] < camarilla_l3_aligned[i] or (jaw_below_teeth and teeth_below_lips):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 (mean reversion failure) or Alligator reverses
            if prices['close'].iloc[i] > camarilla_h3_aligned[i] or (jaw_above_teeth and teeth_above_lips):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if vol_spike[i]:
                # Alligator sleeping: ranging market → mean reversion at H3/L3
                if alligator_sleeping:
                    # Mean reversion long: price touches L3 and Alligator sleeping
                    if prices['close'].iloc[i] <= camarilla_l3_aligned[i] * 1.002:  # 0.2% buffer
                        position = 1
                        signals[i] = 0.25
                    # Mean reversion short: price touches H3 and Alligator sleeping
                    elif prices['close'].iloc[i] >= camarilla_h3_aligned[i] * 0.998:  # 0.2% buffer
                        position = -1
                        signals[i] = -0.25
                # Alligator awakening: trending market → breakout in jaw direction
                else:
                    # Breakout long: jaw above teeth and lips, price breaks H4
                    if jaw_above_teeth and teeth_above_lips and prices['close'].iloc[i] > camarilla_h4_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    # Breakout short: jaw below teeth and lips, price breaks L4
                    elif jaw_below_teeth and teeth_below_lips and prices['close'].iloc[i] < camarilla_l4_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals