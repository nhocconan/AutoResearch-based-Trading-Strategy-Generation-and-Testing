#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams Fractal levels with volume confirmation
# Williams Fractals from 1w provide key reversal points aligned with daily timeframe
# Volume confirmation (current 1d volume > 1.8x 20-period average) filters false breakouts
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# Works in bull/bear: price reacts to weekly structure, volume confirms validity
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "1d_1w_williams_fractal_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams Fractals (5-bar pattern)
    # Bearish fractal: high[n-2] < high[n] and high[n-1] < high[n] and high[n+1] < high[n] and high[n+2] < high[n]
    # Bullish fractal: low[n-2] > low[n] and low[n-1] > low[n] and low[n+1] > low[n] and low[n+2] > low[n]
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    for i in range(2, len(high_1w) - 2):
        if (high_1w[i-2] < high_1w[i] and high_1w[i-1] < high_1w[i] and 
            high_1w[i+1] < high_1w[i] and high_1w[i+2] < high_1w[i]):
            bearish_fractal[i] = high_1w[i]
        if (low_1w[i-2] > low_1w[i] and low_1w[i-1] > low_1w[i] and 
            low_1w[i+1] > low_1w[i] and low_1w[i+2] > low_1w[i]):
            bullish_fractal[i] = low_1w[i]
    
    # Align Williams Fractals to 1d timeframe (need 2-bar extra delay for confirmation)
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x average 1d volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on bearish fractal retracement (mean reversion from resistance)
            if close[i] < bearish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on bullish fractal retracement (mean reversion from support)
            if close[i] > bullish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volume confirmation
            # Long on bullish fractal breakout, Short on bearish fractal breakout
            if volume_confirmed:
                if close[i] > bullish_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < bearish_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals