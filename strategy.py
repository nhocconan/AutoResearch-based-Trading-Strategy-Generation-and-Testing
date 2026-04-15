#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d volume confirmation and 1w trend filter
# Uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish)
# Entry only on volume spike to avoid false signals
# Designed for low trade frequency (target 20-40/year) with clear trend following logic
# Works in both bull (trend continuation) and bear (trend continuation) markets
# Williams Alligator is effective in trending markets and avoids whipsaws in ranging markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for Williams Alligator
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams Alligator components (smoothed moving averages)
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    jaw_raw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward
    teeth_raw = pd.Series(close_4h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    lips_raw = pd.Series(close_4h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (close price)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(close_1w_aligned[i])):
            continue
        
        # Williams Alligator conditions
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Long entry: bullish alignment + price above all lines + volume spike
        if (bullish_alignment and 
            close[i] > lips_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish alignment + price below all lines + volume spike
        elif (bearish_alignment and 
              close[i] < jaws_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite alignment or price crosses middle line (Teeth)
        elif position == 1 and (not bullish_alignment or close[i] < teeth_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish_alignment or close[i] > teeth_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_1dVolume_1wTrend"
timeframe = "4h"
leverage = 1.0