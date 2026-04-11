#!/usr/bin/env python3
# 6h_12h_pivot_volume_v1
# Strategy: 6h price reversal at 12h Camarilla pivot levels with volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance on 12h timeframe.
# Price approaching these levels with diminishing momentum (volume < average) indicates exhaustion.
# Entry: Fade moves toward R3/S3 when volume is below 20-period average (mean reversion).
# Exit: Return to mean (12h VWAP) or opposite pivot level.
# Designed for low frequency (15-30 trades/year) to minimize fee drag in choppy markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (based on previous day)
    # Camarilla: R4 = C + ((H-L) * 1.500), R3 = C + ((H-L) * 1.250)
    #          S3 = C - ((H-L) * 1.250), S4 = C - ((H-L) * 1.500)
    # where C = (H+L+Close)/3 of previous period
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Previous period's values for pivot calculation
    h_prev = np.roll(h_12h, 1)
    l_prev = np.roll(l_12h, 1)
    c_prev = np.roll(c_12h, 1)
    # First value will be invalid (rolled), but we'll handle with valid check
    
    pivot = (h_prev + l_prev + c_prev) / 3.0
    range_prev = h_prev - l_prev
    
    r3 = pivot + (range_prev * 1.250)
    s3 = pivot - (range_prev * 1.250)
    
    # Align pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # 12h VWAP as mean reversion target
    typical_price_12h = (h_12h + l_12h + c_12h) / 3.0
    vwap_12h = (typical_price_12h * df_12h['volume'].values).cumsum() / df_12h['volume'].values.cumsum()
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Volume filter: below average indicates exhaustion
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    low_volume = volume < vol_ma_20  # Volume below 20-period average
    
    # Distance to pivot levels (normalized by ATR-like measure)
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean()
    dist_to_r3 = (r3_aligned - close) / atr_14
    dist_to_s3 = (close - s3_aligned) / atr_14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vwap_12h_aligned[i]) or np.isnan(dist_to_r3[i]) or 
            np.isnan(dist_to_s3[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: price near pivot with low volume (exhaustion)
        near_r3 = dist_to_r3[i] <= 0.5  # Within 0.5x ATR of R3
        near_s3 = dist_to_s3[i] <= 0.5  # Within 0.5x ATR of S3
        
        # Fade R3 (sell pressure exhaustion -> long)
        if near_r3 and low_volume[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Fade S3 (buy pressure exhaustion -> short)
        elif near_s3 and low_volume[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to mean (VWAP) or reaches opposite pivot
        elif position == 1 and (close[i] >= vwap_12h_aligned[i] or close[i] <= s3_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= vwap_12h_aligned[i] or close[i] >= r3_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals