# -*- coding: utf-8 -*-
# -*- mode: python; -*-

#!/usr/bin/env python3
"""
Hypothesis:
6h timeframe with 12h trend filter and 1d volume spike confirmation.
Combines trend-following (price above/below 12h EMA50) with mean-reversion
within 1d range (buy near 1d low, sell near 1d high) during high-volume
breakout conditions. This aims to capture momentum while avoiding false
breakouts in choppy markets. Designed to work in both bull (trend following)
and bear (mean reversion within range) regimes.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_EMA50_VolumeSpike_RangeMeanRev"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for range calculation and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d range (high-low) for mean reversion reference
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    # Use 1d close as reference point for mean reversion
    ref_1d = close_1d
    
    # 1d volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h timeframe
    ema50_12h_6h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ref_1d_6h = align_htf_to_ltf(prices, df_1d, ref_1d)
    range_1d_6h = align_htf_to_ltf(prices, df_1d, range_1d)
    vol_avg_1d_6h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_6h[i]) or np.isnan(ref_1d_6h[i]) or 
            np.isnan(range_1d_6h[i]) or np.isnan(vol_avg_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_12h_6h[i]
        ref_price = ref_1d_6h[i]
        price_range = range_1d_6h[i]
        vol_avg = vol_avg_1d_6h[i]
        vol_ok = volume[i] > vol_avg * 2.0  # Require significant volume spike
        
        # Calculate position within 1d range (0 = at low, 1 = at high)
        if price_range > 0:
            pos_in_range = (close[i] - (ref_price - price_range/2)) / price_range
            pos_in_range = np.clip(pos_in_range, 0, 1)  # Clamp to [0,1]
        else:
            pos_in_range = 0.5  # Avoid division by zero
        
        if position == 0:
            # Look for mean reversion opportunities during high volume
            # Long: near 1d low (0-0.2 range) with volume spike and above trend
            # Short: near 1d high (0.8-1.0 range) with volume spike and below trend
            if vol_ok:
                if pos_in_range < 0.2 and close[i] > trend:
                    signals[i] = 0.25
                    position = 1
                elif pos_in_range > 0.8 and close[i] < trend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long position management
            # Exit: price reaches upper half of range OR trend reverses
            if pos_in_range > 0.6 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position management
            # Exit: price reaches lower half of range OR trend reverses
            if pos_in_range < 0.4 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals