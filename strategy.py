#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1-day Williams Fractal reversal signal
# Williams %R measures overbought/oversold conditions (0 to -100).
# Williams Fractal identifies potential reversal points (bearish fractal = sell signal, bullish fractal = buy signal).
# Strategy: Enter long when Williams %R < -80 (oversold) and bullish fractal confirmed on daily.
# Enter short when Williams %R > -20 (overbought) and bearish fractal confirmed on daily.
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
# Uses 1-day Williams Fractal for reversal confirmation, which requires 2 extra bars for confirmation.
# Designed to capture reversals in both bull and bear markets via overbought/oversold + fractal confirmation.
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Williams Fractal
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractal: bearish (sell) and bullish (buy)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-1] and high[n+1] < high[n-1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-1] and low[n+1] > low[n-1]
    n1d = len(high_1d)
    bearish_fractal = np.zeros(n1d, dtype=bool)
    bullish_fractal = np.zeros(n1d, dtype=bool)
    
    for i in range(2, n1d - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and 
            high_1d[i-3] < high_1d[i-1] and 
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = True
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and 
            low_1d[i-3] > low_1d[i-1] and 
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = True
    
    # Williams Fractal needs 2 extra daily bars for confirmation (after the center bar)
    bearish_fractal_confirmed = bearish_fractal.astype(float)
    bullish_fractal_confirmed = bullish_fractal.astype(float)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_confirmed, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_confirmed, additional_delay_bars=2)
    
    # Calculate 6h Williams %R (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        bull_fract = bullish_fractal_aligned[i]
        bear_fract = bearish_fractal_aligned[i]
        
        if position == 0:
            # Enter long: oversold + bullish fractal
            if wr < -80 and bull_fract > 0.5:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought + bearish fractal
            elif wr > -20 and bear_fract > 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses back above -50
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses back below -50
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1d_Fractal_Reversal"
timeframe = "6h"
leverage = 1.0