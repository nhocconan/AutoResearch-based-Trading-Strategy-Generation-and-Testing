#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Fractal breakouts with volume confirmation
# Williams Fractals from 1d provide swing high/low structure aligned with 12h timeframe
# Volume confirmation (current 12h volume > 1.5x 20-period average) filters false breakouts
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# Works in bull/bear: price reacts to 1d swing structure, volume confirms validity
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "12h_1d_williams_fractal_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Williams Fractals (5-bar pattern)
    # Bearish fractal: high[n-2] < high[n] and high[n-1] < high[n] and high[n+1] < high[n] and high[n+2] < high[n]
    # Bullish fractal: low[n-2] > low[n] and low[n-1] > low[n] and low[n+1] > low[n] and low[n+2] > low[n]
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i-2] < high_1d[i] and high_1d[i-1] < high_1d[i] and 
            high_1d[i+1] < high_1d[i] and high_1d[i+2] < high_1d[i]):
            bearish_fractal[i] = high_1d[i]  # Swing high
        if (low_1d[i-2] > low_1d[i] and low_1d[i-1] > low_1d[i] and 
            low_1d[i+1] > low_1d[i] and low_1d[i+2] > low_1d[i]):
            bullish_fractal[i] = low_1d[i]   # Swing low
    
    # Align Williams Fractals to 12h timeframe with 2-bar extra delay for confirmation
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on retracement to recent bullish fractal (support)
            if close[i] < bullish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on retracement to recent bearish fractal (resistance)
            if close[i] > bearish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volume confirmation
            # Long on bearish fractal breakout (above resistance), Short on bullish fractal breakout (below support)
            if volume_confirmed:
                if close[i] > bearish_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < bullish_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals