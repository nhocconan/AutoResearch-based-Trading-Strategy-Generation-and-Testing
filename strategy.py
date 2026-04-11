#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-day Williams Fractal breakouts + volume confirmation.
# Uses daily Williams Fractals (bearish/bullish) from previous completed day to identify
# structural breakout points. Long when price breaks above bearish fractal with volume
# > 1.5x average, short when breaks below bullish fractal. Designed for low trade
# frequency (~15-35/year) to minimize fee decay while capturing structural breaks.
# Works in bull/bear markets by trading breakouts of key daily support/resistance levels.

name = "6h_1d_williams_fractal_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractal: bearish = high[n-2] < high[n-1] > high[n] > high[n+1] > high[n+2]
    #               bullish  = low[n-2] > low[n-1] < low[n] < low[n+1] < low[n+2]
    n1d = len(high_1d)
    bearish_fractal = np.full(n1d, np.nan)
    bullish_fractal = np.full(n1d, np.nan)
    
    for i in range(2, n1d - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i+1] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i+1] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align daily fractal levels to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate daily average volume (for confirmation)
    volume_1d = df_1d['volume'].values
    vol_avg_10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_10_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 10 to ensure volume average is valid
    for i in range(10, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Entry conditions: price breaks fractal levels with volume
        long_entry = (high[i] > bearish_fractal_aligned[i]) and vol_filter
        short_entry = (low[i] < bullish_fractal_aligned[i]) and vol_filter
        
        # Exit conditions: price returns to opposite fractal level
        exit_long = low[i] < bullish_fractal_aligned[i] if not np.isnan(bullish_fractal_aligned[i]) else False
        exit_short = high[i] > bearish_fractal_aligned[i] if not np.isnan(bearish_fractal_aligned[i]) else False
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals