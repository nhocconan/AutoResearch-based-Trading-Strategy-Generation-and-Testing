#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with Elder Ray force index and volume confirmation.
Long when green line > red line (bullish alignment) and bull power > 0 with volume spike.
Short when red line > green line (bearish alignment) and bear power < 0 with volume spike.
Exit when lines cross or power signals reverse.
Williams Alligator identifies trend direction and alignment; Elder Ray measures bull/bear power;
volume spike confirms institutional participation. Designed for low trade frequency by requiring
trend alignment and power confirmation. Works in both bull and bear markets by following the
12-hour trend with Elder Ray filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for Williams Alligator (13,8,5 SMAs of median price)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_1w = (df_1w['high'].values + df_1w['low'].values) / 2.0
    jaw_1w = pd.Series(median_1w).rolling(window=13, min_periods=13).mean().values
    teeth_1w = pd.Series(median_1w).rolling(window=8, min_periods=8).mean().values
    lips_1w = pd.Series(median_1w).rolling(window=5, min_periods=5).mean().values
    
    # Align to 12h timeframe
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Elder Ray force index: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # Use 1-day data for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 12h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(jaw_1w_aligned[i]) or np.isnan(teeth_1w_aligned[i]) or np.isnan(lips_1w_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator alignment: Lips > Teeth > Jaw = bullish, Jaw > Teeth > Lips = bearish
        bullish_alignment = (lips_1w_aligned[i] > teeth_1w_aligned[i] > jaw_1w_aligned[i])
        bearish_alignment = (jaw_1w_aligned[i] > teeth_1w_aligned[i] > lips_1w_aligned[i])
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_30[i]
        
        if position == 0:
            # Long: Bullish alignment + bull power > 0 + volume spike
            if bullish_alignment and (bull_power_1d_aligned[i] > 0) and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + bear power < 0 + volume spike
            elif bearish_alignment and (bear_power_1d_aligned[i] < 0) and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alignment change or power signal reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Bearish alignment or bull power <= 0
                if not bullish_alignment or (bull_power_1d_aligned[i] <= 0):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bullish alignment or bear power >= 0
                if not bearish_alignment or (bear_power_1d_aligned[i] >= 0):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0