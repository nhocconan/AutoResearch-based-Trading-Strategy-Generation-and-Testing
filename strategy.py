#!/usr/bin/env python3
# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation
# Long when bullish fractal breaks above with 1d EMA50 uptrend and volume > 1.5x average
# Short when bearish fractal breaks below with 1d EMA50 downtrend and volume > 1.5x average
# Exit when price crosses the 20-period EMA
# Williams Fractals identify natural support/resistance; EMA50 filters trend; volume confirms conviction
# Designed for medium-frequency, high-conviction trades on 6h timeframe
# Target: 80-120 total trades over 4 years (20-30/year) with size 0.25

name = "6h_WilliamsFractal_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Fractals
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-1] < low[n+1]
    n_high = len(high)
    bearish_fractal = np.zeros(n_high, dtype=bool)
    bullish_fractal = np.zeros(n_high, dtype=bool)
    
    for i in range(2, n_high - 2):
        # Bearish fractal (peak)
        if (high[i-2] < high[i] and high[i-1] < high[i] and 
            high[i+1] < high[i] and high[i+2] < high[i]):
            bearish_fractal[i] = True
        # Bullish fractal (valley)
        if (low[i-2] > low[i] and low[i-1] > low[i] and 
            low[i+1] > low[i] and low[i+2] > low[i]):
            bullish_fractal[i] = True
    
    # Convert to price levels (0 where no fractal)
    bearish_level = np.where(bearish_fractal, high, 0)
    bullish_level = np.where(bullish_fractal, low, 0)
    
    # For alignment, we need arrays where non-fractal points carry last valid value
    # But since fractals are rare, we'll use the raw levels and align them
    # Williams fractals need 2-bar confirmation delay (already built into calculation above)
    bearish_aligned = align_ltf_to_htf(prices, df_1d, bearish_level)
    bullish_aligned = align_ltf_to_htf(prices, df_1d, bullish_level)
    
    # EMA50 alignment (no extra delay needed as it's based on closed daily candle)
    ema_50_aligned = align_ltf_to_htf(prices, df_1d, ema_50)
    
    # EMA20 for exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(bearish_aligned[i]) or 
            np.isnan(bullish_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price crosses above bullish fractal level with uptrend and volume
            if (bullish_aligned[i] > 0 and close[i] > bullish_aligned[i] and 
                close[i] > ema_50_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below bearish fractal level with downtrend and volume
            elif (bearish_aligned[i] > 0 and close[i] < bearish_aligned[i] and 
                  close[i] < ema_50_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA20
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA20
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals