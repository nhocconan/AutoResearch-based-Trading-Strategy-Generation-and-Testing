#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_VolumeChop
Hypothesis: Camarilla R3/S3 breakouts with volume confirmation and choppiness regime filter. Uses tighter breakout levels (R3/S3) for stronger momentum confirmation. Volume filter ensures trades occur during high participation, reducing false breakouts. Choppiness filter avoids trading in ranging markets (CHOP > 61.8) and only takes trades in trending environments (CHOP < 38.2). Fixed position size 0.25 to control trade frequency and fees. Target: 20-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC for Camarilla levels (R3/S3 = stronger breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (stronger breakout levels)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_1d_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume regime: volume > 70th percentile of 50-period lookback (high volume days only)
    vol_series = pd.Series(volume)
    vol_percentile_70 = vol_series.rolling(window=50, min_periods=50).quantile(0.70).values
    volume_regime = volume > vol_percentile_70
    
    # Choppiness regime: CHOP < 38.2 = trending (only trade in trending markets)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (max(high)-min(low)))
    # Simplified: use rolling std dev of returns as proxy for chop
    returns = np.diff(np.log(close), prepend=0)
    chop = 100 * np.sqrt(np.abs(returns)).rolling(window=14, min_periods=14).mean() / \
           (np.max(high) - np.min(low)).rolling(window=14, min_periods=14).max()
    chop = np.nan_to_num(chop, nan=100.0)  # fill NaN with high chop (no trade)
    chop_regime = chop < 38.2  # only trade when chop < 38.2 (trending)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for volume percentile, 14 for chop)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_percentile_70[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        vol_regime = volume_regime[i]
        chop_reg = chop_regime[i]
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R3/S3 with volume regime AND chop regime (trending only)
        long_entry = (close_val > camarilla_r3_val) and vol_regime and chop_reg
        short_entry = (close_val < camarilla_s3_val) and vol_regime and chop_reg
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_VolumeChop"
timeframe = "4h"
leverage = 1.0