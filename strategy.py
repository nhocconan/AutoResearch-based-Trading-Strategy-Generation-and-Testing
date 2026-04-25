#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trendless periods when lines intertwine.
In 12h timeframe, we trade only when Alligator is "awake" (lines separated) and aligned with 1d EMA50 trend.
Volume confirmation (>2.0x 20-bar vol MA) filters false breakouts. Designed for 12-37 trades/year to minimize fee drag.
Works in bull (long when green alignment) and bear (short when red alignment) markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:  # Need 13 for JAW
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h
    close_12h = pd.Series(df_12h['close'])
    # JAW: 13-period SMMA, 8-period shift
    jaw_12h = close_12h.rolling(window=13, min_periods=13).mean().shift(8).values
    # TEETH: 8-period SMMA, 5-period shift
    teeth_12h = close_12h.rolling(window=8, min_periods=8).mean().shift(5).values
    # LIPS: 5-period SMMA, 3-period shift
    lips_12h = close_12h.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe (already correct, but use helper for consistency)
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (13+8=21), EMA50, and volume MA
    start_idx = max(35, 20)  # 35 for Alligator safety, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_12h_aligned[i]) or 
            np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        jaw = jaw_12h_aligned[i]
        teeth = teeth_12h_aligned[i]
        lips = lips_12h_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Alligator conditions: lines separated and aligned
        # Green alignment (bullish): Lips > Teeth > Jaw
        bullish_alignment = lips > teeth and teeth > jaw
        # Red alignment (bearish): Lips < Teeth < Jaw
        bearish_alignment = lips < teeth and teeth < jaw
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long entry: bullish Alligator + price above EMA + volume
            if bullish_alignment and price_above_ema and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator + price below EMA + volume
            elif bearish_alignment and price_below_ema and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns red OR price crosses below EMA
            if bearish_alignment or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns green OR price crosses above EMA
            if bullish_alignment or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0