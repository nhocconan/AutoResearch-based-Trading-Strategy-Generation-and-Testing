#!/usr/bin/env python3
# 12h_1d_alligator_trend_v1
# Strategy: 12h Williams Alligator (3 SMAs) with 1d trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Williams Alligator identifies trends when jaws (13-period SMA) are above/below teeth (8-period SMA) and lips (5-period SMA).
# In bull markets: go long when lips > teeth > jaws (bullish alignment).
# In bear markets: go short when lips < teeth < jaws (bearish alignment).
# Uses 1d close > SMA50 for bullish trend filter and < SMA50 for bearish filter to avoid counter-trend trades.
# Volume confirmation: 12h volume > 1.5x 20-period average to ensure institutional participation.
# Designed for low trade frequency (~15-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_alligator_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 12h: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # 1d trend filter: close vs SMA50
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Volume confirmation: 12h volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or \
           np.isnan(sma_50_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # 1d trend filter
        bullish_trend = close_1d[i//12] > sma_50_1d[i//12] if i//12 < len(close_1d) else False
        bearish_trend = close_1d[i//12] < sma_50_1d[i//12] if i//12 < len(close_1d) else False
        
        # Entry conditions
        # Long: Alligator bullish AND 1d bullish trend AND volume confirmation
        if bullish_alignment and bullish_trend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Alligator bearish AND 1d bearish trend AND volume confirmation
        elif bearish_alignment and bearish_trend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite Alligator alignment (trend change)
        elif position == 1 and bearish_alignment:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bullish_alignment:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals