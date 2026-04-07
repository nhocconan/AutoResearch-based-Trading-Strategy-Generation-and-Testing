#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot from 1d with volume confirmation
# Hypothesis: Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout) provide high-probability reversal/continuation points.
# In ranging markets, fade at R3/S3; in trending markets, breakout continuation at R4/S4 with volume confirmation.
# Works in bull via R4 breakouts, in bear via S4 breakdowns and R3/S3 mean reversion.
# Target: 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag.
name = "6h_camarilla_pivot_1d_volume_v1"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Based on previous day's high, low, close
    phigh = df_1d['high'].shift(1).values  # Previous day high
    plow = df_1d['low'].shift(1).values    # Previous day low
    pclose = df_1d['close'].shift(1).values # Previous day close
    
    # Pivot point
    pp = (phigh + plow + pclose) / 3.0
    # Camarilla levels
    r4 = pp + ((phigh - plow) * 1.1 / 2)
    r3 = pp + ((phigh - plow) * 1.1 / 4)
    s3 = pp - ((phigh - plow) * 1.1 / 4)
    s4 = pp - ((phigh - plow) * 1.1 / 2)
    
    # Align to 6h timeframe (previous day's levels are known at 00:00 UTC)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate average volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 to ensure previous day data exists
        # Skip if Camarilla levels not available
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 20-day average volume
        vol_confirm = volume[i] > vol_ma_6h[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (mean reversion target) or breaks below S4 (stop)
            if close[i] <= s3_6h[i] or close[i] < s4_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches R3 (mean reversion target) or breaks above R4 (stop)
            if close[i] >= r3_6h[i] or close[i] > r4_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: breakout above R4 with volume confirmation
            if close[i] > r4_6h[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: breakdown below S4 with volume confirmation
            elif close[i] < s4_6h[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
            # Enter long: mean reversion from S3 (oversold bounce)
            elif close[i] < s3_6h[i] and vol_confirm and i > 1 and close[i-1] >= s3_6h[i-1]:
                position = 1
                signals[i] = 0.25
            # Enter short: mean reversion from R3 (overbought rejection)
            elif close[i] > r3_6h[i] and vol_confirm and i > 1 and close[i-1] <= r3_6h[i-1]:
                position = -1
                signals[i] = -0.25
    
    return signals