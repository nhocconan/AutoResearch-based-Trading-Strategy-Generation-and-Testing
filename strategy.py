#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d OHLC, with volume spike confirmation
# Fade at R3/S3 levels (mean reversion in range), breakout continuation at R4/S4 (trend)
# Works in both bull/bear: mean reversion in range, trend following in breakouts
# Volume spike filters weak moves. Target: 25-35 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using yesterday's close, high, low for today's levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla multipliers
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion fade at S3/R3 (range-bound behavior)
        # Long when price touches/slightly below S3 with volume
        if (close[i] <= s3_aligned[i] * 1.002 and  # Allow small buffer
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short when price touches/slightly above R3 with volume
        elif (close[i] >= r3_aligned[i] * 0.998 and  # Allow small buffer
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Breakout continuation at S4/R4 (trend behavior)
        # Long breakout above R4 with volume
        elif (close[i] > r4_aligned[i] and 
              volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short breakdown below S4 with volume
        elif (close[i] < s4_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "6h_Camarilla_R3S3_R4S4_Volume"
timeframe = "6h"
leverage = 1.0