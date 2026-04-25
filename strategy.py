#!/usr/bin/env python3
"""
6h_Williams_Alligator_1dTrend_Filter
Hypothesis: Williams Alligator (jaw/teeth/lips) on 6h identifies market structure (sleeping/awakening/trending). Combined with 1d EMA50 trend filter and volume confirmation, it captures sustained moves in both bull and bear markets. Alligator avoids whipsaws in sideways markets by requiring aligned MAs. Discrete position sizing (0.25) minimizes fee drag. Target: 15-35 trades/year.
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
    
    # Get 6h data for Alligator calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price
    median_6h = (high_6h + low_6h) / 2
    jaw = pd.Series(median_6h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_6h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_6h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to lower timeframe (prices)
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-bar average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Alligator (13+8=21) and EMA50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Alligator alignment: check if MAs are properly ordered (trending) or tangled (sleeping)
        # Bullish alignment: Lips > Teeth > Jaw (all rising)
        # Bearish alignment: Lips < Teeth < Jaw (all falling)
        bullish_align = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_align = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Volume confirmation: current volume > 1.8x 20-bar average
        volume_confirm = volume[i] > 1.8 * vol_ma20[i]
        
        if position == 0:
            # Long: bullish alignment + price above Alligator + uptrend (1d EMA50) + volume
            long_signal = bullish_align and (close[i] > lips_aligned[i]) and (close[i] > ema50_1d_aligned[i]) and volume_confirm
            # Short: bearish alignment + price below Alligator + downtrend (1d EMA50) + volume
            short_signal = bearish_align and (close[i] < lips_aligned[i]) and (close[i] < ema50_1d_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when Alligator starts to sleep (teeth crosses below lips) OR price closes below jaw
            exit_signal = (teeth_aligned[i] < lips_aligned[i]) or (close[i] < jaw_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when Alligator starts to sleep (teeth crosses above lips) OR price closes above jaw
            exit_signal = (teeth_aligned[i] > lips_aligned[i]) or (close[i] > jaw_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Williams_Alligator_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0