#!/usr/bin/env python3
"""
4h Williams Fractal Breakout with Volume Spike and 1d Trend Filter
Long: Bullish fractal breakout above resistance + volume > 2x 4h volume SMA(20) + price > 1d EMA(50)
Short: Bearish fractal breakdown below support + volume > 2x 4h volume SMA(20) + price < 1d EMA(50)
Exit: Price retests the fractal level or opposite breakout
Uses Williams fractals from daily chart for structural levels, volume confirmation, and trend filter
Target: 20-30 trades/year per symbol (80-120 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals and EMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams fractals on 1d data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n] < low[n+2]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    # Need at least 5 points for fractal calculation
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal (peak)
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i-1] < high_1d[i] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i+1] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal (trough)
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i-1] > low_1d[i] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i+1] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams fractals need 2 extra bars for confirmation (as per rule 2b)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d := df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume SMA(20) for volume filter
    vol_sma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 50)  # need EMA50 and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_confirmed[i]) or np.isnan(bullish_fractal_confirmed[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h[i]
        ema_50_val = ema_50_1d_aligned[i]
        bear_fractal = bearish_fractal_confirmed[i]
        bull_fractal = bullish_fractal_confirmed[i]
        
        if position == 0:
            # Long: Bullish fractal breakout + volume > 2x SMA + price > 1d EMA50
            if (not np.isnan(bull_fractal) and 
                price > bull_fractal and 
                close[i-1] <= bull_fractal and 
                vol > 2.0 * vol_sma_val and 
                price > ema_50_val):
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal breakdown + volume > 2x SMA + price < 1d EMA50
            elif (not np.isnan(bear_fractal) and 
                  price < bear_fractal and 
                  close[i-1] >= bear_fractal and 
                  vol > 2.0 * vol_sma_val and 
                  price < ema_50_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price retests fractal level or breaks below bearish fractal
            if (not np.isnan(bull_fractal) and price <= bull_fractal) or \
               (not np.isnan(bear_fractal) and price < bear_fractal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price retests fractal level or breaks above bullish fractal
            if (not np.isnan(bear_fractal) and price >= bear_fractal) or \
               (not np.isnan(bull_fractal) and price > bull_fractal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_VolumeSpike_1dEMA50"
timeframe = "4h"
leverage = 1.0