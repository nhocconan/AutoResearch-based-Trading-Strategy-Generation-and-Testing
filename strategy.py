#!/usr/bin/env python3
"""
6h_1d_williams_fractal_breakout_v1
Hypothesis: Williams Fractals on 1d identify support/resistance; breakouts with volume continuation work in both bull and bear markets.
- Bullish fractal (lowest low with higher lows on both sides) = support
- Bearish fractal (highest high with lower highs on both sides) = resistance
- Breakout above bearish fractal resistance with volume > 1.5x average = long
- Breakdown below bullish fractal support with volume > 1.5x average = short
- Volume filter reduces false breakouts; fractals provide structure in ranging markets
Target: 50-150 total trades over 4 years = 12-37/year
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

name = "6h_1d_williams_fractal_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for fractals
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    
    # Williams fractals need 2 extra 1d bars after center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume average (20-period) for confirmation
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if fractal data not available
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(volume_avg[i])):
            if position != 0:
                # Hold position
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below bullish fractal (support)
            if close[i] < bullish_fractal_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above bearish fractal (resistance)
            if close[i] > bearish_fractal_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * volume_avg[i]
            
            # Long entry: price closes above bearish fractal (resistance) with volume
            if close[i] > bearish_fractal_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below bullish fractal (support) with volume
            elif close[i] < bullish_fractal_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals