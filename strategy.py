#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
# Uses 1d Camarilla pivot levels (R3, R4, S3, S4) calculated from prior day's OHLC
# Enters long on break above R4 with volume > 1.5x 20-period 6h average
# Enters short on break below S4 with volume > 1.5x 20-period 6h average
# Exits when price returns to R3 (for longs) or S3 (for shorts)
# Works in both bull/bear: Camarilla levels adapt to volatility, breakouts capture momentum
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) to minimize fee drag

name = "6h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on prior day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior day's range
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Use prior day's OHLC (i-1) to calculate today's levels
        prior_high = high_1d[i-1]
        prior_low = low_1d[i-1]
        prior_close = close_1d[i-1]
        range_val = prior_high - prior_low
        
        camarilla_r4[i] = prior_close + range_val * 1.1 / 2
        camarilla_r3[i] = prior_close + range_val * 1.1 / 4
        camarilla_s3[i] = prior_close - range_val * 1.1 / 4
        camarilla_s4[i] = prior_close - range_val * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (only use completed daily bars)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 20-period average on 6h (~5 days)
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_6h[i]) or 
            np.isnan(camarilla_r4_6h[i]) or 
            np.isnan(camarilla_s3_6h[i]) or 
            np.isnan(camarilla_s4_6h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back below R3 (take profit at first resistance level)
            if close[i] < camarilla_r3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back above S3 (take profit at first support level)
            if close[i] > camarilla_s3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            # Enter long: price breaks above R4 with volume confirmation
            if (close[i] > camarilla_r4_6h[i] and 
                vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below S4 with volume confirmation
            elif (close[i] < camarilla_s4_6h[i] and 
                  vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals