#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Uses Williams Alligator (Jaw, Teeth, Lips) to identify trend direction and entry points
# Long when Lips > Teeth > Jaw (bullish alignment) and price above 1d EMA50 with volume confirmation
# Short when Lips < Teeth < Jaw (bearish alignment) and price below 1d EMA50 with volume confirmation
# Exit when Alligator alignment breaks or price crosses 1d EMA50 in opposite direction
# Position size: 0.25 to balance return and drawdown
# Williams Alligator is effective in both trending and ranging markets, suitable for 2025's expected conditions

name = "12h_WilliamsAlligator_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 12h: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # Using SMMA (Smoothed Moving Average) approximation with EMA
    jaw = pd.Series(close).ewm(span=13, adjust=False).mean().ewm(span=8, adjust=False).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False).mean().ewm(span=5, adjust=False).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False).mean().ewm(span=3, adjust=False).mean().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.3 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Alligator components
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bullish Alligator alignment (Lips > Teeth > Jaw) + price above 1d EMA50 + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish Alligator alignment (Lips < Teeth < Jaw) + price below 1d EMA50 + volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks OR price crosses below 1d EMA50
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks OR price crosses above 1d EMA50
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals