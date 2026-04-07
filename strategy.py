#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot Reversal with Volume Filter
# Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) act as strong support/resistance on 6h timeframe.
# Price rejecting R3/S3 with volume confirmation indicates reversal opportunity.
# Works in bull (buy S3 bounce) and bear (sell R3 rejection). Volume filter reduces false breaks.
# Target: 15-30 trades/year to minimize fee drag on 6h.
name = "6h_camarilla_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using (H+L+C)/3 as pivot point approximation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    # Range
    rng = prev_high - prev_low
    
    # Camarilla levels
    r3 = pp + (rng * 1.1 / 2)
    s3 = pp - (rng * 1.1 / 2)
    r4 = pp + (rng * 1.1)
    s4 = pp - (rng * 1.1)
    
    # Align to 6h timeframe (already shift(1) in get_htf_data, but we shifted again for prev day)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 6h volume > 20-period EMA of volume
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if pivot data not available
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(vol_ema[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > vol_ema[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 or reaches R4 (take profit)
            if close[i] < s3_6h[i] or close[i] > r4_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above R3 or reaches S4 (take profit)
            if close[i] > r3_6h[i] or close[i] < s4_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Enter long: price rejects S3 with volume (close > S3 and volume confirmation)
            if close[i] > s3_6h[i] and vol_confirm:
                # Additional confirmation: not breaking below S4 (avoid strong breakdown)
                if close[i] > s4_6h[i]:
                    position = 1
                    signals[i] = 0.25
            # Enter short: price rejects R3 with volume (close < R3 and volume confirmation)
            elif close[i] < r3_6h[i] and vol_confirm:
                # Additional confirmation: not breaking above R4 (avoid strong breakout)
                if close[i] < r4_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals