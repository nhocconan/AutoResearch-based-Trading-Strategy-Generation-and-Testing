#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1d EMA trend filter and volume confirmation
# Uses Williams Fractals to identify potential reversal points, filters by 1-day EMA trend,
# and requires volume surge for confirmation. Works in both bull and bear markets by
# only taking breakouts in the direction of the higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Fractals and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Fractals (5-bar window: 2 bars each side)
    # Bearish fractal: high is highest of 5 bars
    # Bullish fractal: low is lowest of 5 bars
    n1 = len(high_1d)
    bearish_fractal = np.full(n1, np.nan)
    bullish_fractal = np.full(n1, np.nan)
    
    for i in range(2, n1 - 2):
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Calculate EMA34 on 1d for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Williams Fractals need 2 extra bars for confirmation (per rule 2b)
    bearish_fractal_confirmed = bearish_fractal.copy()
    bullish_fractal_confirmed = bullish_fractal.copy()
    
    # Align indicators to 4h timeframe with extra delay for fractals
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_confirmed, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_confirmed, additional_delay_bars=2)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            continue
        
        # Long entry: bullish fractal breakout + volume surge + price above EMA34
        if (not np.isnan(bullish_fractal_aligned[i]) and
            close[i] > bullish_fractal_aligned[i] and
            volume[i] > 2.0 * vol_avg_1d_aligned[i] and
            close[i] > ema34_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish fractal breakout + volume surge + price below EMA34
        elif (not np.isnan(bearish_fractal_aligned[i]) and
              close[i] < bearish_fractal_aligned[i] and
              volume[i] > 2.0 * vol_avg_1d_aligned[i] and
              close[i] < ema34_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal
        elif position == 1 and close[i] < ema34_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema34_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsFractal_EMA34_Volume"
timeframe = "4h"
leverage = 1.0