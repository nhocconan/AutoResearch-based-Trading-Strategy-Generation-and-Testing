#!/usr/bin/env python3
"""
12h_Williams_Fractal_Breakout_With_Volume_Confirmation
Hypothesis: Williams fractals on 1d identify key reversal points. Breakouts above bearish fractals or below bullish fractals with volume confirmation capture momentum in both bull and bear markets. Low-frequency signals reduce fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams fractals on daily data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n+1] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n+1] < low[n+2]
    n1 = len(high_1d)
    bearish_fractal = np.full(n1, np.nan)
    bullish_fractal = np.full(n1, np.nan)
    
    for i in range(2, n1 - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i-1] > high_1d[i] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i+1] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i-1] < low_1d[i] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i+1] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align fractals to 12h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume spike: >1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above bearish fractal with volume spike
            if not np.isnan(bearish_fractal_val) and price > bearish_fractal_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below bullish fractal with volume spike
            elif not np.isnan(bullish_fractal_val) and price < bullish_fractal_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price falls below bullish fractal (trend reversal)
            if not np.isnan(bullish_fractal_val) and price < bullish_fractal_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price rises above bearish fractal (trend reversal)
            if not np.isnan(bearish_fractal_val) and price > bearish_fractal_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Williams_Fractal_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0