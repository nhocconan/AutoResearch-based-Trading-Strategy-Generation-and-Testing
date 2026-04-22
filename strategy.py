#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with volume spike and 1d EMA34 trend filter.
# Long when bullish fractal breaks above previous high + volume spike + price > 1d EMA34
# Short when bearish fractal breaks below previous low + volume spike + price < 1d EMA34
# Exit when price crosses back through fractal level or volume drops below 80% of average.
# Williams Fractals identify potential turning points, effective in both trending and ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    # Load 1d data for fractal calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals (5-bar pattern)
    # Bullish fractal: low[n-2] < low[n-1] and low[n] < low[n-1] and low[n+1] < low[n-1] and low[n+2] < low[n-1]
    # Bearish fractal: high[n-2] > high[n-1] and high[n] > high[n-1] and high[n+1] > high[n-1] and high[n+2] > high[n-1]
    n_1d = len(high_1d)
    bullish_fractal = np.full(n_1d, np.nan)
    bearish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        if (low_1d[i-2] > low_1d[i] and 
            low_1d[i-1] > low_1d[i] and 
            low_1d[i+1] > low_1d[i] and 
            low_1d[i+2] > low_1d[i]):
            bullish_fractal[i] = low_1d[i]  # Store the low value
        
        if (high_1d[i-2] < high_1d[i] and 
            high_1d[i-1] < high_1d[i] and 
            high_1d[i+1] < high_1d[i] and 
            high_1d[i+2] < high_1d[i]):
            bearish_fractal[i] = high_1d[i]  # Store the high value
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if (np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        ema34 = ema34_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above bullish fractal + volume spike + price > EMA34
            if (not np.isnan(bullish_fractal_val) and 
                price > bullish_fractal_val and 
                vol_spike and 
                price > ema34):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below bearish fractal + volume spike + price < EMA34
            elif (not np.isnan(bearish_fractal_val) and 
                  price < bearish_fractal_val and 
                  vol_spike and 
                  price < ema34):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through fractal level or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below bullish fractal or volume dries up
                if (not np.isnan(bullish_fractal_val) and price < bullish_fractal_val) or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above bearish fractal or volume dries up
                if (not np.isnan(bearish_fractal_val) and price > bearish_fractal_val) or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0