#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Williams Fractals (bearish/bullish) as dynamic support/resistance
# - Bearish fractal (sell signal): highest high with two lower highs on each side
# - Bullish fractal (buy signal): lowest low with two higher lows on each side
# - Enters long when price closes above a bullish fractal with volume confirmation
# - Enters short when price closes below a bearish fractal with volume confirmation
# - Exits when price returns to the midline between the last two fractals of opposite type
# - Williams fractals are confirmed only after 2 additional candles, so we use additional_delay_bars=2
# - Designed to work in both bull and bear markets by adapting to natural support/resistance levels
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_WilliamsFractal_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals (5-bar pattern)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    # Williams Fractal: point is fractal if it's the highest/lowest in 5-bar window
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: highest high with two lower highs on each side
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal: lowest low with two higher lows on each side
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams fractals require 2 additional bars for confirmation
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume filters (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track last bullish and bearish fractal levels for exit logic
    last_bullish = np.nan
    last_bearish = np.nan
    
    for i in range(100, n):  # Start after warmup
        # Update last confirmed fractal levels
        if not np.isnan(bullish_fractal_confirmed[i]):
            last_bullish = bullish_fractal_confirmed[i]
        if not np.isnan(bearish_fractal_confirmed[i]):
            last_bearish = bearish_fractal_confirmed[i]
        
        # Skip if any critical value is NaN
        if (np.isnan(bullish_fractal_confirmed[i]) and np.isnan(bearish_fractal_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for bullish breakout: price closes above bullish fractal with volume spike
            bullish_breakout = (not np.isnan(last_bullish) and 
                              close[i] > last_bullish and 
                              volume_spike[i])
            
            # Look for bearish breakout: price closes below bearish fractal with volume spike
            bearish_breakout = (not np.isnan(last_bearish) and 
                              close[i] < last_bearish and 
                              volume_spike[i])
            
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint between last bullish and bearish fractal
            if (not np.isnan(last_bullish) and not np.isnan(last_bearish) and last_bearish < last_bullish):
                midpoint = (last_bullish + last_bearish) / 2
                if close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Default exit: price closes below the bullish fractal
                if close[i] < last_bullish:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint between last bullish and bearish fractal
            if (not np.isnan(last_bullish) and not np.isnan(last_bearish) and last_bearish < last_bullish):
                midpoint = (last_bullish + last_bearish) / 2
                if close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Default exit: price closes above the bearish fractal
                if close[i] > last_bearish:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals