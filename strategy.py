#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h EMA50 trend filter and volume confirmation.
# Long when bullish fractal forms + price > 12h EMA50 + volume > 1.5x 20-period average.
# Short when bearish fractal forms + price < 12h EMA50 + volume > 1.5x 20-period average.
# Exit when price closes back inside the previous fractal's opposite level.
# Williams fractals require 2-bar confirmation, so we add 2-bar delay when aligning.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_Williams_Fractal_Breakout_12hEMA50_Volume"
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
    
    # Get 12h data for trend filter and volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume moving average for volume filter
    vol_ma_20 = df_12h['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Get 1d data for Williams fractals (requires daily candles for proper formation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full_like(high_1d, np.nan)
    bullish_fractal = np.full_like(low_1d, np.nan)
    
    # Calculate fractals (need at least 5 points: n-2 to n+2)
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: middle bar is highest
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and 
            high_1d[i-1] > high_1d[i-3] and 
            high_1d[i-1] > high_1d[i+1]):
            bearish_fractal[i-1] = high_1d[i-1]  # Value at the fractal point
        
        # Bullish fractal: middle bar is lowest
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and 
            low_1d[i-1] < low_1d[i-3] and 
            low_1d[i-1] < low_1d[i+1]):
            bullish_fractal[i-1] = low_1d[i-1]  # Value at the fractal point
    
    # Williams fractals need 2-bar confirmation after the center bar
    # So we add 2-bar delay when aligning to ensure the fractal is confirmed
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or \
           np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 1.5x 20-period average
        # Find the most recent completed 12h bar
        idx_12h = 0
        while idx_12h < len(df_12h) and df_12h.iloc[idx_12h]['open_time'] <= prices.iloc[i]['open_time']:
            idx_12h += 1
        idx_12h -= 1  # last completed 12h bar
        
        if idx_12h < 0:
            vol_filter = False
        else:
            vol_12h_current = df_12h.iloc[idx_12h]['volume']
            vol_filter = vol_12h_current > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
        
        if position == 0:
            # Look for entry: Williams fractal breakout + trend + volume
            long_condition = (bullish_fractal_aligned[i] > 0 and 
                             close[i] > ema_50_aligned[i] and 
                             vol_filter)
            short_condition = (bearish_fractal_aligned[i] > 0 and 
                              close[i] < ema_50_aligned[i] and 
                              vol_filter)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes back below the bullish fractal level
            if close[i] <= bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes back above the bearish fractal level
            if close[i] >= bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals