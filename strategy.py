#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume spike confirmation.
- Primary timeframe: 12h for lower trade frequency (target 50-150 total trades over 4 years).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on median price.
- Entry: Long when Alligator aligned bullish (Lips > Teeth > Jaw) AND price > Lips AND 1d EMA50 bullish AND volume spike.
         Short when Alligator aligned bearish (Lips < Teeth < Jaw) AND price < Lips AND 1d EMA50 bearish AND volume spike.
- Exit: Price crosses opposite Alligator line (Lips crosses Jaw) OR loss of volume confirmation.
- Volume: Current 12h volume > 1.5 * 20-period volume MA.
- Signal size: 0.25 discrete to minimize fee churn.
- Target: 80-120 total trades over 4 years (20-30/year) for 12h timeframe.
Williams Alligator identifies trending vs ranging markets. In strong trends, the lines are well-separated and aligned.
Volume spike confirms institutional participation. EMA50 filter ensures we only trade with the daily trend.
This combination should work in both bull and bear markets by capturing strong directional moves with confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 1d volume MA for volume confirmation
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams Alligator components (SMA of median price)
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    # Align HTF indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (max shift is 8 for jaw)
    start_idx = 13  # Need at least 13 for jaw calculation + 8 shift = 21, but we use 13 to be safe
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or np.isnan(lips_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_lips = lips_values[i]
        curr_jaw = jaw_values[i]
        
        # Alligator alignment conditions
        bullish_aligned = (curr_lips > teeth_values[i]) and (teeth_values[i] > jaw_values[i])
        bearish_aligned = (curr_lips < teeth_values[i]) and (teeth_values[i] < jaw_values[i])
        
        if position == 0:
            # Check for entry signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish entry: Alligator bullish AND price > Lips AND 1d EMA50 bullish
                if bullish_aligned and curr_close > curr_lips and curr_close > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Alligator bearish AND price < Lips AND 1d EMA50 bearish
                elif bearish_aligned and curr_close < curr_lips and curr_close < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below Jaw (Alligator wake up signal) OR loss of volume confirmation
            if curr_close < curr_jaw or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Jaw (Alligator wake up signal) OR loss of volume confirmation
            if curr_close > curr_jaw or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0