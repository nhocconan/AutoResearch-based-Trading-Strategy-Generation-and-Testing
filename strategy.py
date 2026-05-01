#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly pivot direction and volume confirmation
# Williams Fractals identify potential turning points. Weekly pivot provides HTF bias.
# Volume spike confirms breakout validity. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull (breakouts with weekly pivot alignment) and bear (fade at extremes with confirmation).

name = "6h_WilliamsFractal_Breakout_WeeklyPivot_VolumeSpike_v1"
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
    
    # Weekly HTF data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly pivot points (using prior completed weekly bar)
    # Pivot = (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot for each weekly bar (using prior bar to avoid look-ahead)
    weekly_pivot = np.full(len(weekly_close), np.nan)
    for i in range(1, len(weekly_close)):
        weekly_pivot[i] = (weekly_high[i-1] + weekly_low[i-1] + weekly_close[i-1]) / 3.0
    
    # Align weekly pivot to 6h timeframe (waits for weekly bar close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Williams Fractals on 6h data (requires 2 extra bars for confirmation)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n+1] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n+1] < low[n+2]
    bearish_fractal = np.full(len(high), np.nan)
    bullish_fractal = np.full(len(low), np.nan)
    
    for i in range(2, len(high)-2):
        # Bearish fractal (peak)
        if (high[i-2] < high[i-1] and 
            high[i-1] < high[i] and 
            high[i] > high[i+1] and 
            high[i+1] > high[i+2]):
            bearish_fractal[i] = high[i]
        
        # Bullish fractal (trough)
        if (low[i-2] > low[i-1] and 
            low[i-1] > low[i] and 
            low[i] < low[i+1] and 
            low[i+1] < low[i+2]):
            bullish_fractal[i] = low[i]
    
    # Align fractals with additional 2-bar delay for confirmation (as per Rule 2b)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, prices, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, prices, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 20 for volume MA + extra for fractals
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        weekly_pivot = weekly_pivot_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish fractal break above weekly pivot with volume spike
            if (not np.isnan(bullish_fractal_aligned[i]) and 
                curr_close > weekly_pivot and 
                curr_low <= bullish_fractal_aligned[i] and  # Price touched fractal level
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal break below weekly pivot with volume spike
            elif (not np.isnan(bearish_fractal_aligned[i]) and 
                  curr_close < weekly_pivot and 
                  curr_high >= bearish_fractal_aligned[i] and  # Price touched fractal level
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below weekly pivot or opposite fractal
            if curr_close < weekly_pivot or not np.isnan(bearish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above weekly pivot or opposite fractal
            if curr_close > weekly_pivot or not np.isnan(bullish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals