#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume spike.
# Long when green line > red line (bullish alignment) with 1d uptrend and volume spike (>2x avg).
# Short when green line < red line (bearish alignment) with 1d downtrend and volume spike.
# Uses Williams Alligator (SMAs: Jaw=13, Teeth=8, Lips=5) on 4h for trend alignment.
# Designed for ~20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following 1d trend and requiring volatility expansion.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Williams Alligator lines on 4h: Jaw (13), Teeth (8), Lips (5)
    # All are SMAs with specific periods
    jaw_4h = pd.Series(close_4h).rolling(window=13, min_periods=13).mean().values
    teeth_4h = pd.Series(close_4h).rolling(window=8, min_periods=8).mean().values
    lips_4h = pd.Series(close_4h).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 4h timeframe (wait for 4h bar to close)
    jaw_4h_aligned = align_htf_to_ltf(prices, df_4h, jaw_4h)
    teeth_4h_aligned = align_htf_to_ltf(prices, df_4h, teeth_4h)
    lips_4h_aligned = align_htf_to_ltf(prices, df_4h, lips_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_4h_aligned[i]) or np.isnan(teeth_4h_aligned[i]) or 
            np.isnan(lips_4h_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = (lips_4h_aligned[i] > teeth_4h_aligned[i]) and (teeth_4h_aligned[i] > jaw_4h_aligned[i])
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = (lips_4h_aligned[i] < teeth_4h_aligned[i]) and (teeth_4h_aligned[i] < jaw_4h_aligned[i])
        
        # Long conditions: bullish alignment AND 1d uptrend AND volume spike
        if (bullish_alignment and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: bearish alignment AND 1d downtrend AND volume spike
        elif (bearish_alignment and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0