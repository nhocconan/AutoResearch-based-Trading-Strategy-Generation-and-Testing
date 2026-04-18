#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator system with Elder Ray force confirmation
# Uses 3 smoothed moving averages (jaw/teeth/lips) for trend direction
# Elder Ray bull/bear power confirms trend strength with volume weighting
# Works in bull (teeth above jaw, bull power positive) and bear (teeth below jaw, bear power negative)
# Designed for 12-37 trades/year to avoid fee drag
name = "12h_Alligator_ElderRay_Force_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator: 3 SMAs of median price
    median_price = (df_1d['high'] + df_1d['low']) / 2
    
    # Jaw: 13-period SMMA, teeth: 8-period, lips: 5-period
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    jaw = median_price.ewm(alpha=1/13, adjust=False).mean().values
    teeth = median_price.ewm(alpha=1/8, adjust=False).mean().values
    lips = median_price.ewm(alpha=1/5, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    ema13 = df_1d['close'].ewm(span=13, adjust=False).mean().values
    bull_power = (df_1d['high'] - ema13).values
    bear_power = (ema13 - df_1d['low']).values
    
    # Align to 12h timeframe (wait for daily close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        
        if position == 0:
            # Long: Alligator aligned up (lips > teeth > jaw) AND bull power positive
            if lips_val > teeth_val > jaw_val and bull_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down (lips < teeth < jaw) AND bear power positive
            elif lips_val < teeth_val < jaw_val and bear_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks OR bull power turns negative
            if not (lips_val > teeth_val > jaw_val) or bull_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks OR bear power turns negative
            if not (lips_val < teeth_val < jaw_val) or bear_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals