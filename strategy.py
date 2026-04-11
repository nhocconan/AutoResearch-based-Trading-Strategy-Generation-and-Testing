#!/usr/bin/env python3
"""
6h/1d Fractal Breakout with Volume Confirmation
- Uses Williams Fractals on 1d to identify key support/resistance
- Breakout above/below fractal levels with volume confirmation
- Designed for 50-150 trades over 4 years (12-37/year)
- Works in bull/bear: breakouts capture momentum in any regime
"""

from typing import Tuple
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

name = "6h_fractal_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate Williams Fractals on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Fractals need 2 extra bars for confirmation (center + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume filter: 24-period average (4 days worth of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.8x 24-period average
        volume_filter = volume[i] > 1.8 * vol_ma_24[i]
        
        # Breakout conditions using Fractal levels
        breakout_up = close[i] > bullish_fractal_aligned[i]  # Break above bullish fractal (resistance)
        breakdown_down = close[i] < bearish_fractal_aligned[i]  # Break below bearish fractal (support)
        
        # Entry conditions
        long_entry = breakout_up and volume_filter
        short_entry = breakdown_down and volume_filter
        
        # Exit conditions: return to opposite fractal level
        long_exit = close[i] < bearish_fractal_aligned[i]  # Return below bearish fractal
        short_exit = close[i] > bullish_fractal_aligned[i]  # Return above bullish fractal
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals