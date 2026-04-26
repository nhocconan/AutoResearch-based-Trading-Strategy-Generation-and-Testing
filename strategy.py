#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_VolumeSpike_DynamicRegime
Hypothesis: Camarilla R3/S3 breakouts with volume spike (>2x median volume) and dynamic regime filter (CHOP > 50 = ranging, CHOP < 30 = trending). Uses opposing logic: in ranging markets (CHOP>50) fade breakouts (short at R3, long at S3); in trending markets (CHOP<30) follow breakouts (long at R3, short at S3). Volume spike ensures participation. Fixed size 0.25 to limit trades. Target: 20-30 trades/year. Works in both bull and bear via regime adaptation.
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
    
    # Load 12h data ONCE before loop for HTF filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Previous 12h bar's OHLC for Camarilla levels (R3/S3 = stronger breakout levels)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_vals = df_12h['close'].values
    
    # Calculate Camarilla levels: R3, S3 (stronger breakout levels)
    rng = high_12h - low_12h
    camarilla_r3 = close_12h_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_12h_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume spike: volume > 2x 50-period median volume (adaptive to volatility)
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=50, min_periods=50).median().values
    volume_spike = volume > (2.0 * vol_median)
    
    # Choppiness regime: CHOP > 50 = ranging, CHOP < 30 = trending
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / np.log(highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_ranging = chop > 50   # ranging market: fade breakouts
    chop_trending = chop < 30  # trending market: follow breakouts
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for volume median, 14 for ATR)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_median[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        chop_range = chop_ranging[i]
        chop_trend = chop_trending[i]
        size = fixed_size
        
        # Entry conditions: volume spike required
        if not vol_spike:
            signals[i] = 0.0 if position == 0 else (size if position == 1 else -size)
            continue
        
        # Regime-dependent logic
        if chop_range:
            # Ranging market: fade breakouts (mean reversion)
            long_entry = close_val < camarilla_s3_val  # short at R3, long at S3 (fade)
            short_entry = close_val > camarilla_r3_val
        elif chop_trend:
            # Trending market: follow breakouts (momentum)
            long_entry = close_val > camarilla_r3_val  # long at R3, short at S3 (follow)
            short_entry = close_val < camarilla_s3_val
        else:
            # Neutral chop: no trade
            long_entry = False
            short_entry = False
        
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
            # Long - exit on mean reversion to midpoint (Camarilla center) in ranging, or break of S3 in trending
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if chop_range:
                # In ranging: exit at midpoint
                if close_val < mid_point:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            else:
                # In trending: exit if price breaks below S3 (failed breakout)
                if close_val < camarilla_s3_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint in ranging, or break of R3 in trending
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if chop_range:
                # In ranging: exit at midpoint
                if close_val > mid_point:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            else:
                # In trending: exit if price breaks above R3 (failed breakout)
                if close_val > camarilla_r3_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_VolumeSpike_DynamicRegime"
timeframe = "4h"
leverage = 1.0