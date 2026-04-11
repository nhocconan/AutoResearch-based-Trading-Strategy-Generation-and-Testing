#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Strategy: Camarilla pivot levels from daily chart with volume spike confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price reacting at Camarilla pivot levels (S3/S4 for longs, R3/R4 for shorts)
# with volume spike indicates institutional interest. Works in bull by catching bounces
# at support and in bear by catching rejections at resistance. Volume filters false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's high, low, close
    phigh = df_1d['high'].shift(1).values  # Previous day high
    plow = df_1d['low'].shift(1).values    # Previous day low
    pclose = df_1d['close'].shift(1).values # Previous day close
    
    # Camarilla levels
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1 / 4)
    # S3 = close - ((high - low) * 1.1 / 4)
    # S4 = close - ((high - low) * 1.1 / 2)
    r4 = pclose + ((phigh - plow) * 1.1 / 2)
    r3 = pclose + ((phigh - plow) * 1.1 / 4)
    s3 = pclose - ((phigh - plow) * 1.1 / 4)
    s4 = pclose - ((phigh - plow) * 1.1 / 2)
    
    # Align to 12h timeframe (wait for daily bar to close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike detection (20-period average)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Require 2x average volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price = close[i]
        
        # Long conditions: price at or below S3/S4 with volume spike
        long_signal = ((price <= s3_aligned[i] * 1.002) or (price <= s4_aligned[i] * 1.002)) and vol_spike[i]
        
        # Short conditions: price at or above R3/R4 with volume spike
        short_signal = ((price >= r3_aligned[i] * 0.998) or (price >= r4_aligned[i] * 0.998)) and vol_spike[i]
        
        # Exit conditions: price moves back toward midpoint (pivot)
        pivot = (r3_aligned[i] + s3_aligned[i]) / 2
        exit_long = position == 1 and price >= pivot
        exit_short = position == -1 and price <= pivot
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals