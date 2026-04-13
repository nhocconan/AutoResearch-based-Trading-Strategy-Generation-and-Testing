#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator uses three SMAs (Jaw, Teeth, Lips) to identify trends and avoid whipsaws.
# In bull markets: Lips > Teeth > Jaw indicates uptrend.
# In bear markets: Lips < Teeth < Jaw indicates downtrend.
# Volume confirmation ensures trend has participation.
# 1d trend filter avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 12h data
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    # Using EMA as proxy for SMMA for computational efficiency
    jaw = np.full(n, np.nan)
    teeth = np.full(n, np.nan)
    lips = np.full(n, np.nan)
    
    # Calculate SMMA-like values using EMA with appropriate periods
    jaw_raw = pd.Series(close).ewm(span=13, adjust=False).mean().values
    teeth_raw = pd.Series(close).ewm(span=8, adjust=False).mean().values
    lips_raw = pd.Series(close).ewm(span=5, adjust=False).mean().values
    
    # Apply shifts (note: shift forward in time, so we use negative indices)
    for i in range(8, n):
        jaw[i] = jaw_raw[i-8]
    for i in range(5, n):
        teeth[i] = teeth_raw[i-5]
    for i in range(3, n):
        lips[i] = lips_raw[i-3]
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 1d trend filter: EMA(50) on daily data
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(50, len(close_1d)):
        ema_1d[i] = pd.Series(close_1d[:i+1]).ewm(span=50, adjust=False).mean().iloc[-1]
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_1d_val = ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume + price above 1d EMA
            if (lips[i] > teeth[i] and 
                teeth[i] > jaw[i] and 
                volume_confirm and
                price > ema_1d_val):
                position = 1
                signals[i] = position_size
            # Short: Lips < Teeth < Jaw (bearish alignment) + volume + price below 1d EMA
            elif (lips[i] < teeth[i] and 
                  teeth[i] < jaw[i] and 
                  volume_confirm and
                  price < ema_1d_val):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator lines intertwine (Lips crosses Teeth or Jaw)
            if (lips[i] < teeth[i] or 
                lips[i] < jaw[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator lines intertwine (Lips crosses Teeth or Jaw)
            if (lips[i] > teeth[i] or 
                lips[i] > jaw[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Williams_Alligator_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0