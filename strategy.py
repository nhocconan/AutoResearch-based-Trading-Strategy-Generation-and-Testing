#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 trend + volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) from 6h for trend structure and entry timing
# 1d EMA50 provides higher timeframe trend filter to reduce whipsaw vs shorter TF
# Volume confirmation (>1.3x 20 EMA) filters low-participation false signals
# Session filter (08-20 UTC) to avoid low-liquidity periods
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# Works in both bull and bear: Alligator adapts to trending/choppy markets via convergence/divergence.

name = "6h_WilliamsAlligator_1dEMA50_VolumeConfirm_Session"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 6h data for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h: SMA with specific offsets
    # Jaw: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward  
    # Lips: 5-period SMA, shifted 3 bars forward
    close_6h = pd.Series(df_6h['close'])
    jaw_6h = close_6h.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_6h = close_6h.rolling(window=8, min_periods=8).mean().shift(5).values
    lips_6h = close_6h.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align 6h Alligator components to primary 6h timeframe (completed 6h bar only)
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw_6h)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth_6h)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips_6h)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) + price above EMA50 + volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema50_aligned[i] and 
                volume[i] > (1.3 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Jaw > Teeth > Lips (bearish alignment) + price below EMA50 + volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and 
                  close[i] < ema50_aligned[i] and 
                  volume[i] > (1.3 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines converge (Lips < Teeth) OR price below EMA50 OR weak volume
            if (lips_aligned[i] < teeth_aligned[i] or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines converge (Jaw < Teeth) OR price above EMA50 OR weak volume
            if (jaw_aligned[i] < teeth_aligned[i] or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals