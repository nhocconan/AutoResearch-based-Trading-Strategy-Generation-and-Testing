#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Williams Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) smoothed with future offset.
# In uptrend: Lips > Teeth > Jaw; in downtrend: Lips < Teeth < Jaw.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades.
# Volume spike (>1.5x 20-period average) confirms institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Three SMAs with future offset (smoothed)
    # Jaw: 13-period SMA, smoothed by 8 bars
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values  # Smoothed by 8
    
    # Teeth: 8-period SMA, smoothed by 5 bars
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values  # Smoothed by 5
    
    # Lips: 5-period SMA, smoothed by 3 bars
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values  # Smoothed by 3
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trending market logic with Alligator alignment and volume filter
        if close[i] > ema50_1d_aligned[i] and volume_filter[i]:  # Uptrend filter
            # Bullish when Lips > Teeth > Jaw (Alligator aligned up)
            if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                signals[i] = 0.25
                position = 1
            # Exit long when Alligator alignment breaks
            elif position == 1 and not (lips[i] > teeth[i] and teeth[i] > jaw[i]):
                signals[i] = 0.0
                position = 0
        elif close[i] < ema50_1d_aligned[i] and volume_filter[i]:  # Downtrend filter
            # Bearish when Lips < Teeth < Jaw (Alligator aligned down)
            if lips[i] < teeth[i] and teeth[i] < jaw[i]:
                signals[i] = -0.25
                position = -1
            # Exit short when Alligator alignment breaks
            elif position == -1 and not (lips[i] < teeth[i] and teeth[i] < jaw[i]):
                signals[i] = 0.0
                position = 0
        else:
            # Hold current position or stay flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeFilter"
timeframe = "12h"
leverage = 1.0