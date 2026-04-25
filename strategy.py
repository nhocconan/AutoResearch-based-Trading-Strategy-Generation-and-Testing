#!/usr/bin/env python3
"""
1d_WilliamsAlligator_Trend_HTF1wFilter_v1
Hypothesis: Use Williams Alligator on 1d to identify trend direction (jaw/teeth/lips alignment). 
Only trade in direction of 1w HTF trend (price > EMA50 on weekly). 
Enter on Alligator bullish/bearish alignment with price confirmation. 
Exit on opposite Alligator signal or HTF trend reversal.
Designed for low frequency (target 15-25 trades/year) to minimize fee drag in bear markets.
Works in bull via trend following, works in bear via HTF filter avoiding counter-trend trades.
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
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1d: SMAs of median price
    # Median price = (high + low) / 2
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid
    
    # Align Alligator lines to LTF (1d -> 1d is identity, but we use for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1w HTF trend: EMA50 on weekly close
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Alligator (max shift 8 + jaw period 13 = 21) and 1w EMA50 (50)
    start_idx = max(21, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine Alligator alignment
        # Bullish: lips > teeth > jaw (green alignment)
        alligator_bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish: jaws > teeth > lips (red alignment)
        alligator_bearish = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Determine 1w HTF trend
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long setup: Alligator bullish + 1w uptrend
            long_setup = alligator_bullish and htf_1w_bullish
            # Short setup: Alligator bearish + 1w downtrend
            short_setup = alligator_bearish and htf_1w_bearish
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Alligator turns bearish OR 1w trend turns bearish
            if (not alligator_bullish) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Alligator turns bullish OR 1w trend turns bullish
            if (not alligator_bearish) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WilliamsAlligator_Trend_HTF1wFilter_v1"
timeframe = "1d"
leverage = 1.0