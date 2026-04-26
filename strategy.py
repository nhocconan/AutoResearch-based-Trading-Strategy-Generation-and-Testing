#!/usr/bin/env python3
"""
12h_Williams_Alligator_R1_S1_Camarilla_v1
Hypothesis: Williams Alligator (jaw/teeth/lips) on 1w as trend filter, combined with Camarilla R1/S1 levels on 1d for mean-reversion entries in ranging markets and breakout entries in trending markets. Volume spike confirms. Works in both bull and bear regimes by adapting to market structure via Alligator alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels on 1d: based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Previous day's values (shift by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_open = np.roll(open_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    prev_open[0] = np.nan
    
    # Camarilla R1, S1, R3, S3 levels
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.0 / 12
    S1 = prev_close - camarilla_range * 1.0 / 12
    R3 = prev_close + camarilla_range * 3.0 / 12
    S3 = prev_close - camarilla_range * 3.0 / 12
    
    # Load 1w data ONCE before loop for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    
    # Williams Alligator on 1w: SMAs of median price
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    median_price_1w = (high_1w + low_1w) / 2
    
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    
    # Lips: 5-period SMMA shifted 3 bars
    lips = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    
    # Alligator alignment: 
    # Bullish: Lips > Teeth > Jaw (green alignment)
    # Bearish: Lips < Teeth < Jaw (red alignment)
    # Otherwise: ranging/market chop
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Align Alligator components and Camarilla levels to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    bullish_alignment_aligned = align_htf_to_ltf(prices, df_1w, bullish_alignment.astype(float))
    bearish_alignment_aligned = align_htf_to_ltf(prices, df_1w, bearish_alignment.astype(float))
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(13, 8, 5, 20) + 8  # Alligator needs 13 + 8 shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(bullish_alignment_aligned[i]) or
            np.isnan(bearish_alignment_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine market regime from Alligator
        bullish_regime = bullish_alignment_aligned[i] > 0.5
        bearish_regime = bearish_alignment_aligned[i] > 0.5
        ranging_regime = not (bullish_regime or bearish_regime)
        
        # Long logic
        long_signal = False
        if bullish_regime:
            # In bullish trend: buy on dip to S1 with volume spike
            if close[i] <= S1_aligned[i] and volume_spike[i]:
                long_signal = True
        elif bearish_regime:
            # In bearish trend: buy on break above R3 (potential reversal) with volume spike
            if close[i] > R3_aligned[i] and volume_spike[i]:
                long_signal = True
        else:
            # In ranging market: buy at S1 with volume spike (mean reversion)
            if close[i] <= S1_aligned[i] and volume_spike[i]:
                long_signal = True
        
        # Short logic
        short_signal = False
        if bullish_regime:
            # In bullish trend: sell on break below S3 (potential reversal) with volume spike
            if close[i] < S3_aligned[i] and volume_spike[i]:
                short_signal = True
        elif bearish_regime:
            # In bearish trend: sell on rally to R1 with volume spike
            if close[i] >= R1_aligned[i] and volume_spike[i]:
                short_signal = True
        else:
            # In ranging market: sell at R1 with volume spike (mean reversion)
            if close[i] >= R1_aligned[i] and volume_spike[i]:
                short_signal = True
        
        # Exit logic: opposite signal or regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: short signal generated OR regime turns bearish
            if short_signal or bearish_regime:
                exit_long = True
        elif position == -1:
            # Exit short: long signal generated OR regime turns bullish
            if long_signal or bullish_regime:
                exit_short = True
        
        # Update signals and position
        if exit_long:
            signals[i] = 0.0
            position = 0
        elif exit_short:
            signals[i] = 0.0
            position = 0
        elif long_signal:
            signals[i] = 0.25
            position = 1
        elif short_signal:
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_R1_S1_Camarilla_v1"
timeframe = "12h"
leverage = 1.0