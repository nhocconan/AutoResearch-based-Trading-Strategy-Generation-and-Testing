#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_WeeklyTrend_Volume"
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
    
    # Get weekly data for Williams fractal and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Williams fractal from weekly high/low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Bearish fractal: high[n-2] < high[n-1] and high[n] < high[n-1]
    # Bullish fractal: low[n-2] > low[n-1] and low[n] > low[n-1]
    bearish_fractal = np.zeros(len(high_1w), dtype=bool)
    bullish_fractal = np.zeros(len(low_1w), dtype=bool)
    
    for i in range(2, len(high_1w)-2):
        if (high_1w[i-2] < high_1w[i-1] and 
            high_1w[i] < high_1w[i-1] and
            high_1w[i+1] < high_1w[i-1] and
            high_1w[i+2] < high_1w[i-1]):
            bearish_fractal[i] = True
            
        if (low_1w[i-2] > low_1w[i-1] and 
            low_1w[i] > low_1w[i-1] and
            low_1w[i+1] > low_1w[i-1] and
            low_1w[i+2] > low_1w[i-1]):
            bullish_fractal[i] = True
    
    # Align fractals to 6h with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Weekly trend: EMA34 of weekly close
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = align_htf_to_ltf(prices, df_1d, (vol_1d > vol_avg_1d * 2.0).astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_6h[i]) or np.isnan(vol_spike_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: daily volume spike
        vol_spike = vol_spike_1d[i] > 0.5
        
        if position == 0:
            # Long: Bullish fractal + above weekly EMA34 + volume spike
            if (bullish_fractal_aligned[i] > 0.5 and 
                close[i] > ema34_6h[i] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal + below weekly EMA34 + volume spike
            elif (bearish_fractal_aligned[i] > 0.5 and 
                  close[i] < ema34_6h[i] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish fractal OR price below weekly EMA34
            if (bearish_fractal_aligned[i] > 0.5 or 
                close[i] < ema34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish fractal OR price above weekly EMA34
            if (bullish_fractal_aligned[i] > 0.5 or 
                close[i] > ema34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals