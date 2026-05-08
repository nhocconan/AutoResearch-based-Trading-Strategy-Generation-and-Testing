#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Williams Fractal for reversal signals with volume confirmation.
# Uses weekly bearish/bullish fractals to identify potential reversal points in higher timeframe structure.
# Long when bullish fractal forms on weekly chart and price closes above fractal high with volume confirmation.
# Short when bearish fractal forms on weekly chart and price closes below fractal low with volume confirmation.
# Exit when price moves back to the fractal level or opposite fractal forms.
# Designed for low trade frequency (10-20/year) to avoid fee decay. Works in trending and ranging markets via fractal structure.

name = "1d_1wWilliamsFractal_Reversal"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Williams Fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Williams Fractals (5-bar pattern)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] and high[n+1] < high[n]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-2] and low[n+1] > low[n]
    n_1w = len(high_1w)
    bearish_fractal = np.zeros(n_1w, dtype=bool)
    bullish_fractal = np.zeros(n_1w, dtype=bool)
    
    for i in range(2, n_1w - 2):
        # Bearish fractal: middle bar is highest
        if (high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i-2] and
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: middle bar is lowest
        if (low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i-2] and
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = True
    
    # Convert to float arrays for alignment (1.0 at fractal points, 0.0 otherwise)
    bearish_fractal_float = bearish_fractal.astype(float)
    bullish_fractal_float = bullish_fractal.astype(float)
    
    # Need 2-bar confirmation for fractals (as per Williams Fractal rules)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal_float, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal_float, additional_delay_bars=2)
    
    # Store actual fractal levels for entry/exit
    bearish_level = np.full(n_1w, np.nan)
    bullish_level = np.full(n_1w, np.nan)
    bearish_level[bearish_fractal] = high_1w[bearish_fractal]
    bullish_level[bullish_fractal] = low_1w[bullish_fractal]
    
    bearish_level_aligned = align_htf_to_ltf(prices, df_1w, bearish_level)
    bullish_level_aligned = align_htf_to_ltf(prices, df_1w, bullish_level)
    
    # Volume confirmation: daily volume > 1.5x 20-day EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_level_aligned[i]) and np.isnan(bullish_level_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish fractal formed and price closes above fractal low with volume
            if not np.isnan(bullish_level_aligned[i]) and bullish_level_aligned[i] > 0:
                if close[i] > bullish_level_aligned[i] and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: bearish fractal formed and price closes below fractal high with volume
            elif not np.isnan(bearish_level_aligned[i]) and bearish_level_aligned[i] > 0:
                if close[i] < bearish_level_aligned[i] and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to fractal level or bearish fractal forms
            if not np.isnan(bullish_level_aligned[i]) and bullish_level_aligned[i] > 0:
                if close[i] <= bullish_level_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif not np.isnan(bearish_level_aligned[i]) and bearish_level_aligned[i] > 0:
                # Bearish fractal formed - exit long
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to fractal level or bullish fractal forms
            if not np.isnan(bearish_level_aligned[i]) and bearish_level_aligned[i] > 0:
                if close[i] >= bearish_level_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif not np.isnan(bullish_level_aligned[i]) and bullish_level_aligned[i] > 0:
                # Bullish fractal formed - exit short
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals