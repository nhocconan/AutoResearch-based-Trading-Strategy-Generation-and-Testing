#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Alligator uses smoothed MAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# In bull: Lips > Teeth > Jaw (green), buy on alignment + volume.
# In bear: Lips < Teeth < Jaw (red), sell on alignment + volume.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume filter avoids false signals. Target: 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for Alligator components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d close for Alligator SMAs
    close_1d = pd.Series(df_1d['close'].values)
    
    # Alligator: Jaw (13-period SMMA), Teeth (8-period), Lips (5-period)
    # SMMA = smoothed moving average (similar to EMA but different smoothing)
    jaw_1d = close_1d.ewm(alpha=1/13, adjust=False).mean().values  # approx SMMA(13)
    teeth_1d = close_1d.ewm(alpha=1/8, adjust=False).mean().values   # approx SMMA(8)
    lips_1d = close_1d.ewm(alpha=1/5, adjust=False).mean().values    # approx SMMA(5)
    
    # Align to 12h timeframe
    jaw_12h = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Weekly EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw (green Alligator)
        bullish_alignment = (lips_12h[i] > teeth_12h[i]) and (teeth_12h[i] > jaw_12h[i])
        # Bearish alignment: Lips < Teeth < Jaw (red Alligator)
        bearish_alignment = (lips_12h[i] < teeth_12h[i]) and (teeth_12h[i] < jaw_12h[i])
        
        # Long conditions: Bullish alignment + price above weekly EMA34 (uptrend) + volume
        if bullish_alignment and (close[i] > ema34_1w_aligned[i]) and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short conditions: Bearish alignment + price below weekly EMA34 (downtrend) + volume
        elif bearish_alignment and (close[i] < ema34_1w_aligned[i]) and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "12h_WilliamsAlligator_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0