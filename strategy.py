#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume spike confirmation.
# Williams Alligator: Jaw (13-period smoothed median), Teeth (8-period smoothed median), Lips (5-period smoothed median).
# In uptrend: Lips > Teeth > Jaw. In downtrend: Lips < Teeth < Jaw.
# 1d EMA50 filter ensures we trade only in the direction of the daily trend.
# Volume spike (>2x 20-period average) confirms conviction.
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator: Smoothed medians (using close as proxy for median price)
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values  # Smoothed median (13)
    teeth = close_s.rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values    # Smoothed median (8)
    lips = close_s.rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values     # Smoothed median (5)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Alligator alignment
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (uptrend alignment) AND uptrend AND volume spike
            if lips_above_teeth and teeth_above_jaw and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (downtrend alignment) AND downtrend AND volume spike
            elif lips_below_teeth and teeth_below_jaw and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks (Lips < Teeth or Teeth < Jaw) OR trend reverses
            if not (lips_above_teeth and teeth_above_jaw) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks (Lips > Teeth or Teeth > Jaw) OR trend reverses
            if not (lips_below_teeth and teeth_below_jaw) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals