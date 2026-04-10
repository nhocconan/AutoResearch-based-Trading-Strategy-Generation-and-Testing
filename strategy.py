#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator (13,8,5 SMAs) from 1d: identifies trend direction and strength
# - 12h price must be outside Alligator's mouth (above lips/teeth/jaw for long, below for short)
# - Volume confirmation: current 12h volume > 1.8x 30-period average to confirm momentum
# - Designed for 12h timeframe: targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: Alligator converges in ranging markets (no signal), diverges in trends
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "12h_1d_alligator_volume_v1"
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
    
    # Pre-compute Williams Alligator (13,8,5) on 1d
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_1d = (high_1d + low_1d) / 2  # Typical price
    
    # Jaw (13-period, shifted 8 bars)
    jaw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:13] = np.nan  # First 13 values invalid due to roll
    
    # Teeth (8-period, shifted 5 bars)
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:8] = np.nan  # First 8 values invalid due to roll
    
    # Lips (5-period, shifted 3 bars)
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:5] = np.nan  # First 5 values invalid due to roll
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_30 = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume_12h > (1.8 * avg_volume_30)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside Alligator's mouth (below lips)
            if prices['close'].iloc[i] < lips_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside Alligator's mouth (above lips)
            if prices['close'].iloc[i] > lips_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator divergence with volume confirmation
            if vol_spike[i]:
                # Alligator is diverging (lips > teeth > jaw for uptrend, lips < teeth < jaw for downtrend)
                lips_val = lips_aligned[i]
                teeth_val = teeth_aligned[i]
                jaw_val = jaw_aligned[i]
                
                # Strong uptrend: lips above teeth above jaw
                if lips_val > teeth_val and teeth_val > jaw_val:
                    # Enter long when price closes above lips (confirming breakout)
                    if prices['close'].iloc[i] > lips_aligned[i]:
                        position = 1
                        entry_price = prices['close'].iloc[i]
                        signals[i] = 0.25
                # Strong downtrend: lips below teeth below jaw
                elif lips_val < teeth_val and teeth_val < jaw_val:
                    # Enter short when price closes below lips (confirming breakout)
                    if prices['close'].iloc[i] < lips_aligned[i]:
                        position = -1
                        entry_price = prices['close'].iloc[i]
                        signals[i] = -0.25
    
    return signals