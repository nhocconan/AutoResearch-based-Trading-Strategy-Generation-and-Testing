#!/usr/bin/env python3
"""
4h Williams Alligator + 1d EMA Trend Filter with Volume Spike
Long: Alligator bullish (Lips > Teeth > Jaw) + price above Teeth + 1d EMA up + volume spike
Short: Alligator bearish (Lips < Teeth < Jaw) + price below Teeth + 1d EMA down + volume spike
Exit: Opposite Alligator alignment or price crosses Jaw
Williams Alligator identifies trend, EMA filter ensures alignment with higher timeframe trend,
volume spike confirms momentum. Designed for 4h timeframe to capture trends in both bull and bear markets.
Target: 80-150 total trades over 4 years (20-38/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs"""
    # Jaw: Blue line - 13-period SMMA shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: Red line - 8-period SMMA shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: Green line - 5-period SMMA shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 4h
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d EMA slope for trend filter
    ema_slope = np.diff(ema_34_1d_aligned, prepend=ema_34_1d_aligned[0])
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # need Alligator calculations
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_slope[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Alligator bullish + price above Teeth + 1d EMA up + volume spike
            if (lips[i] > teeth[i] > jaw[i] and 
                price > teeth[i] and 
                ema_slope[i] > 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + price below Teeth + 1d EMA down + volume spike
            elif (lips[i] < teeth[i] < jaw[i] and 
                  price < teeth[i] and 
                  ema_slope[i] < 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator bearish OR price crosses below Jaw
            if (lips[i] < teeth[i] < jaw[i]) or (price < jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator bullish OR price crosses above Jaw
            if (lips[i] > teeth[i] > jaw[i]) or (price > jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0