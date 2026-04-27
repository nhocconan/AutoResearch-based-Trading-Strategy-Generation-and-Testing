#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation.
# Go long when price breaks above a bearish fractal (resistance) with bullish 1d EMA trend and volume spike.
# Go short when price breaks below a bullish fractal (support) with bearish 1d EMA trend and volume spike.
# Exit when price returns to the 1d EMA34 (mean reversion to trend).
# Williams fractals require 2-bar confirmation after the center bar, so we use additional_delay_bars=2.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing reversals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals: bearish (resistance) and bullish (support)
    # Bearish fractal: high[n] is highest among high[n-2], high[n-1], high[n], high[n+1], high[n+2]
    # Bullish fractal: low[n] is lowest among low[n-2], low[n-1], low[n], low[n+1], low[n+2]
    # Requires 2 bars after center for confirmation -> additional_delay_bars=2
    def calculate_fractals(high_arr, low_arr):
        n = len(high_arr)
        bearish = np.full(n, np.nan)
        bullish = np.full(n, np.nan)
        for i in range(2, n-2):
            if (high_arr[i] >= high_arr[i-1] and high_arr[i] >= high_arr[i-2] and
                high_arr[i] >= high_arr[i+1] and high_arr[i] >= high_arr[i+2]):
                bearish[i] = high_arr[i]
            if (low_arr[i] <= low_arr[i-1] and low_arr[i] <= low_arr[i-2] and
                low_arr[i] <= low_arr[i+1] and low_arr[i] <= low_arr[i+2]):
                bullish[i] = low_arr[i]
        return bearish, bullish
    
    bearish_fractal, bullish_fractal = calculate_fractals(high_1d, low_1d)
    
    # Align fractals to 6h timeframe with additional 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 30-period average (moderate to balance frequency)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above bearish fractal (resistance), above 1d EMA34, volume spike
        if (close[i] > bearish_fractal_aligned[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below bullish fractal (support), below 1d EMA34, volume spike
        elif (close[i] < bullish_fractal_aligned[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to 1d EMA34 (mean reversion to trend)
        elif position == 1 and close[i] < ema34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dEMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0