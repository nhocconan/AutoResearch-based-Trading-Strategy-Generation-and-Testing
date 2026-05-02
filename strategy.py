#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) combination
# Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) defines trend regime and avoids chop
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low measures trend strength
# Entry: Alligator aligned (teeth > lips > jaw for bull, reverse for bear) + Elder Ray confirmation
# Exit: Alligator misalignment or Elder Ray divergence
# Works in bull via trend continuation, bear via trend-following alignment with power filters
# Low frequency: targets 12-30 trades/year on 6h timeframe with 0.25 sizing

name = "6h_WilliamsAlligator_ElderRay_v1"
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
    
    # 1d HTF data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13-period SMA), Teeth (8-period SMA), Lips (5-period SMA)
    close_1d = df_1d['close'].values
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # 1d HTF data for Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = ema_13_1d - low_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for Alligator (13 periods)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator alignment (trend regime)
        bullish_alignment = teeth_1d_aligned[i] > lips_1d_aligned[i] > jaw_1d_aligned[i]
        bearish_alignment = jaw_1d_aligned[i] > lips_1d_aligned[i] > teeth_1d_aligned[i]
        
        # Elder Ray confirmation: trend strength
        strong_bull_power = bull_power_1d_aligned[i] > 0
        strong_bear_power = bear_power_1d_aligned[i] > 0
        
        if position == 0:  # Flat - look for new entries
            if bullish_alignment and strong_bull_power:
                # Long: Alligator bullish alignment + positive Bull Power
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and strong_bear_power:
                # Short: Alligator bearish alignment + positive Bear Power
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0  # Avoid chop or weak trends
        
        elif position == 1:  # Long position
            # Exit: Alligator misalignment or Bear Power becomes stronger
            if not bullish_alignment or bear_power_1d_aligned[i] > bull_power_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator misalignment or Bull Power becomes stronger
            if not bearish_alignment or bull_power_1d_aligned[i] > bear_power_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals