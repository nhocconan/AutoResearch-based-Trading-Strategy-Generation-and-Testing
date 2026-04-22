#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams Fractal Breakout with 1-day Trend Filter and Volume Confirmation.
Long when price breaks above bearish fractal high during 1-day uptrend with volume spike.
Short when price breaks below bullish fractal low during 1-day downtrend with volume spike.
Exit when price returns to the fractal midpoint or trend reverses.
Williams fractals provide natural support/resistance levels that work in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for fractals and trend - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1D data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Align fractals to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 20-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal + 1d uptrend + volume spike
            if close[i] > bearish_fractal_aligned[i] and ema20_1d_aligned[i] > ema20_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal + 1d downtrend + volume spike
            elif close[i] < bullish_fractal_aligned[i] and ema20_1d_aligned[i] < ema20_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to fractal midpoint or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below fractal midpoint or 1d trend turns down
                if bullish_fractal_aligned[i] > 0:  # valid bullish fractal
                    midpoint = (bearish_fractal_aligned[i] + bullish_fractal_aligned[i]) / 2
                    if close[i] < midpoint or ema20_1d_aligned[i] < ema20_1d_aligned[i-1]:
                        exit_signal = True
            else:  # position == -1
                # Exit short: price above fractal midpoint or 1d trend turns up
                if bearish_fractal_aligned[i] > 0:  # valid bearish fractal
                    midpoint = (bearish_fractal_aligned[i] + bullish_fractal_aligned[i]) / 2
                    if close[i] > midpoint or ema20_1d_aligned[i] > ema20_1d_aligned[i-1]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Williams_Fractal_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0