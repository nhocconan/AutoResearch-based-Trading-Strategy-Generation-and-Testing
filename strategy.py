#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Alligator uses SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends and convergence/divergence
# In uptrend: Lips > Teeth > Jaw; Downtrend: Lips < Teeth < Jaw
# Convergence (lines intertwine) = ranging market, divergence = trending
# 1d EMA50 provides higher timeframe trend bias to avoid counter-trend trades
# Volume confirmation filters weak signals
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "12h_WilliamsAlligator_1dEMA50_Volume"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Jaw (13)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Teeth (8)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # Lips (5)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + above 1d EMA50 + volume confirmation
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + below 1d EMA50 + volume confirmation
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator converges (Lips <= Teeth or Teeth <= Jaw) or breaks below 1d EMA50
            if (lips[i] <= teeth[i] or teeth[i] <= jaw[i]) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator converges (Lips >= Teeth or Teeth >= Jaw) or breaks above 1d EMA50
            if (lips[i] >= teeth[i] or teeth[i] >= jaw[i]) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals