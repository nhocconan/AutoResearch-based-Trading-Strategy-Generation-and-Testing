#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend via SMAs with future shifts
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# Volume spike (>1.8x 20-period EMA) filters low-probability signals
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components on 12h timeframe
    # Jaw (Blue): 13-period SMMA shifted 8 bars ahead
    # Teeth (Red): 8-period SMMA shifted 5 bars ahead  
    # Lips (Green): 5-period SMMA shifted 3 bars ahead
    # Using SMA as approximation for SMMA (required for proper calculation)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start from 60 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (balanced for 12h)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Alligator signals with 1d trend filter
        # Long: Lips > Teeth > Jaw (bullish alignment) + price above 1d EMA50 + volume spike
        # Short: Lips < Teeth < Jaw (bearish alignment) + price below 1d EMA50 + volume spike
        if position == 0:
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_50_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips < Teeth or Teeth < Jaw) OR price below 1d EMA50
            if lips[i] < teeth[i] or teeth[i] < jaw[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips > Teeth or Teeth > Jaw) OR price above 1d EMA50
            if lips[i] > teeth[i] or teeth[i] > jaw[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals