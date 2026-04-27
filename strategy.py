#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume spike
# Williams Alligator uses 3 SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# In uptrend: Lips > Teeth > Jaw (green alignment)
# In downtrend: Lips < Teeth < Jaw (red alignment)
# Trend filter uses 1d EMA50 to avoid counter-trend trades.
# Volume spike confirms momentum. Target: 20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator SMAs on 6h data
    # Jaw: 13-period SMA, shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift forward 5 bars
    teeth[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift forward 3 bars
    lips[:3] = np.nan
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Lips > Teeth > Jaw (green alignment) + price above 1d EMA50 + volume
        if (lips[i] > teeth[i] and 
            teeth[i] > jaw[i] and  
            close[i] > ema50_1d_aligned[i] and   # Uptrend filter
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Lips < Teeth < Jaw (red alignment) + price below 1d EMA50 + volume
        elif (lips[i] < teeth[i] and 
              teeth[i] < jaw[i] and  
              close[i] < ema50_1d_aligned[i] and   # Downtrend filter
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "6h_WilliamsAlligator_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0