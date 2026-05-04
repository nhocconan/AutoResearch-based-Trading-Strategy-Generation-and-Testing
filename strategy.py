#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Uses 1w EMA50 for trend direction and Williams Alligator (Jaw/Teeth/Lips) from 1d for entry/exit
# Volume confirmation requires 1.5x average volume to ensure strong participation
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag on 1d timeframe
# Williams Alligator excels in ranging markets (mean reversion) while EMA50 filter ensures trend alignment
# Works in both bull and bear markets by following the 1w trend direction and using Alligator for structure
# Prioritizes BTC/ETH performance with SOL as secondary

name = "1d_WilliamsAlligator_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for Alligator (13,8,5 SMAs)
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator from 1d data
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward  
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Jaw: 13-period SMMA of typical price, shifted 8 bars
    jaw = pd.Series(typical_price_1d).ewm(alpha=1/13, adjust=False).mean().values
    jaw = np.roll(jaw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan  # first 8 values invalid due to shift
    
    # Teeth: 8-period SMMA of typical price, shifted 5 bars
    teeth = pd.Series(typical_price_1d).ewm(alpha=1/8, adjust=False).mean().values
    teeth = np.roll(teeth, 5)  # shift forward 5 bars
    teeth[:5] = np.nan  # first 5 values invalid due to shift
    
    # Lips: 5-period SMMA of typical price, shifted 3 bars
    lips = pd.Series(typical_price_1d).ewm(alpha=1/5, adjust=False).mean().values
    lips = np.roll(lips, 3)  # shift forward 3 bars
    lips[:3] = np.nan  # first 3 values invalid due to shift
    
    # Align Alligator lines to 1d timeframe (already on 1d, but need to align for safety)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Williams Alligator signals with 1w trend filter
        # Alligator sleeping (all lines intertwined) = ranging market
        # Alligator awakening (lines diverging) = trending market
        # Lips above Teeth above Jaw = uptrend (green > red > blue)
        # Lips below Teeth below Jaw = downtrend (green < red < blue)
        if position == 0:
            # Long: Lips above Teeth above Jaw (bullish alignment) + volume spike + price above 1w EMA50 (uptrend)
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                volume_spike and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips below Teeth below Jaw (bearish alignment) + volume spike + price below 1w EMA50 (downtrend)
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines cross (Lips < Teeth or Teeth < Jaw) OR price below 1w EMA50 (trend change)
            if (lips_aligned[i] < teeth_aligned[i] or teeth_aligned[i] < jaw_aligned[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines cross (Lips > Teeth or Teeth > Jaw) OR price above 1w EMA50 (trend change)
            if (lips_aligned[i] > teeth_aligned[i] or teeth_aligned[i] > jaw_aligned[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals