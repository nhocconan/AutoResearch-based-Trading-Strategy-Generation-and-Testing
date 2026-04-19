#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h Williams Fractal breakout and volume confirmation.
# Uses 12h Williams Fractals to identify potential reversal points at swing highs/lows.
# Enters long when price breaks above a bearish fractal (resistance) with volume confirmation,
# and short when price breaks below a bullish fractal (support) with volume confirmation.
# Exit when price returns to the opposite fractal level or on opposing signal.
# Williams Fractals require 2-bar confirmation after the center bar, so we use additional_delay_bars=2.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "6h_12h_WilliamsFractal_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Fractal calculation (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Williams Fractals: bearish (swing high) and bullish (swing low)
    # A bearish fractal is a high with 2 lower highs on each side
    # A bullish fractal is a low with 2 higher lows on each side
    n_12h = len(high_12h)
    bearish_fractal = np.zeros(n_12h, dtype=bool)
    bullish_fractal = np.zeros(n_12h, dtype=bool)
    
    for i in range(2, n_12h - 2):
        if (high_12h[i] > high_12h[i-1] and high_12h[i] > high_12h[i-2] and
            high_12h[i] > high_12h[i+1] and high_12h[i] > high_12h[i+2]):
            bearish_fractal[i] = True
        if (low_12h[i] < low_12h[i-1] and low_12h[i] < low_12h[i-2] and
            low_12h[i] < low_12h[i+1] and low_12h[i] < low_12h[i+2]):
            bullish_fractal[i] = True
    
    # Convert to arrays: value at fractal points, 0 elsewhere
    bearish_values = np.where(bearish_fractal, high_12h, 0.0)
    bullish_values = np.where(bullish_fractal, low_12h, 0.0)
    
    # Align fractal levels to 6h timeframe with additional delay for confirmation
    # Williams Fractals need 2 extra 12h bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_values, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_values, additional_delay_bars=2)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or invalid
        if np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        bearish_level = bearish_fractal_aligned[i]
        bullish_level = bullish_fractal_aligned[i]
        
        # Skip if no valid fractal level (0.0 means no fractal)
        if bearish_level == 0.0 and bullish_level == 0.0:
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long when price breaks above bearish fractal (resistance) with volume
            if bearish_level > 0 and close[i] > bearish_level and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below bullish fractal (support) with volume
            elif bullish_level > 0 and close[i] < bullish_level and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to bullish fractal (support) or on bearish breakout
            if bullish_level > 0 and close[i] < bullish_level:
                signals[i] = 0.0
                position = 0
            elif bearish_level > 0 and close[i] > bearish_level and volume_filter[i]:
                # Reverse to short on new bearish breakout
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to bearish fractal (resistance) or on bullish breakout
            if bearish_level > 0 and close[i] > bearish_level:
                signals[i] = 0.0
                position = 0
            elif bullish_level > 0 and close[i] < bullish_level and volume_filter[i]:
                # Reverse to long on new bullish breakout
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = -0.25
    
    return signals