#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDir_VolumeConfirm_v1
Hypothesis: 6h Donchian(20) breakout in direction of weekly Camarilla pivot (R3/S3) with volume confirmation.
Weekly pivot defines structural support/resistance; breakouts in its direction have higher follow-through.
Volume spike confirms institutional participation. Discrete sizing (0.25) limits fee drag.
Target: 50-150 total trades over 4 years (12-37/year) by requiring HTF alignment, breakout, and volume.
Works in bull/bear: weekly pivot adapts to regime; volume filter avoids false breakouts in ranging markets.
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
    
    # Load weekly data ONCE before loop for HTF pivot
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    # Camarilla: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # We use R3 for longs, S3 for shorts (stronger levels than R1/S1)
    camarilla_r1w = df_1w['close'] + 1.1 * (df_1w['high'] - df_1w['low']) / 12
    camarilla_s1w = df_1w['close'] - 1.1 * (df_1w['high'] - df_1w['low']) / 12
    camarilla_r3w = camarilla_r1w + 2 * camarilla_r1w  # R3 = R2 + 2*(R2-R1) ≈ R1 + 4*(R1-S1) [simplified]
    camarilla_s3w = camarilla_s1w - 2 * (camarilla_r1w - camarilla_s1w)  # S3 = S2 - 2*(R1-S1)
    # Actually: R3 = Close + 1.1*(High-Low)*6/12, S3 = Close - 1.1*(High-Low)*6/12
    camarilla_r3w = df_1w['close'] + 1.1 * (df_1w['high'] - df_1w['low']) * 6 / 12
    camarilla_s3w = df_1w['close'] - 1.1 * (df_1w['high'] - df_1w['low']) * 6 / 12
    
    # Align weekly Camarilla to 6h timeframe
    camarilla_r3w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3w.values)
    camarilla_s3w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3w.values)
    
    # Calculate 6h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for volume MA, 20 for Donchian)
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_30[i]) or np.isnan(camarilla_r3w_aligned[i]) or 
            np.isnan(camarilla_s3w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.8 * vol_ma_30[i]
        
        # Donchian breakout conditions
        breakout_above = close[i] > high_20[i-1]  # Use previous bar's channel to avoid look-ahead
        breakout_below = close[i] < low_20[i-1]
        
        if breakout_above and volume_spike:
            # Long signal: breakout above Donchian high with volume, above weekly S3 (bullish bias)
            if close[i] > camarilla_s3w_aligned[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            else:
                # Hold or flatten if not aligned with weekly pivot
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = 0.0
                    position = 0
        elif breakout_below and volume_spike:
            # Short signal: breakout below Donchian low with volume, below weekly R3 (bearish bias)
            if close[i] < camarilla_r3w_aligned[i]:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold or flatten if not aligned with weekly pivot
                if position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
                    position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotDir_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0