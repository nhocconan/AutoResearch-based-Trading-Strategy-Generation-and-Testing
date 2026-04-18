#!/usr/bin/env python3
"""
4h Williams Fractal Reversal with Volume Spike and 12h EMA34 Filter
Hypothesis: Williams Fractals identify swing highs/lows. A bearish fractal (sell signal) forms when a middle high has two lower highs on each side; bullish fractal (buy signal) forms when a middle low has two higher lows on each side. 
Entry: Bearish fractal confirmed + volume spike + price below 12h EMA34 = short; Bullish fractal confirmed + volume spike + price above 12h EMA34 = long.
Exit: Opposite fractal forms.
Volume filter: current volume > 2.0 x 20-period EMA of volume to avoid low-volume false signals.
Uses 12h EMA34 as trend filter to avoid counter-trend trades in chop.
Designed for 4-8 trades/year per symbol to minimize fee impact.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data once for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: volume > 2.0 x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema20
    
    # Williams Fractals on price (requires 5-bar window: t-2, t-1, t, t+1, t+2)
    # Bearish fractal: high[t] > high[t-2] and high[t] > high[t-1] and high[t] > high[t+1] and high[t] > high[t+2]
    # Bullish fractal: low[t] < low[t-2] and low[t] < low[t-1] and low[t] < low[t+1] and low[t] < low[t+2]
    bearish_fractal = np.zeros(n, dtype=bool)
    bullish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        if (high[i] > high[i-2] and high[i] > high[i-1] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish_fractal[i] = True
        if (low[i] < low[i-2] and low[i] < low[i-1] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish_fractal[i] = True
    
    # Fractals need 2-bar confirmation after the center bar (the pattern completes at t+2)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, pd.DataFrame({'index': np.arange(n)}), bearish_fractal.astype(float), additional_delay_bars=2) > 0.5
    bullish_fractal_confirmed = align_htf_to_ltf(prices, pd.DataFrame({'index': np.arange(n)}), bullish_fractal.astype(float), additional_delay_bars=2) > 0.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for fractals (need 2 bars after) + EMA/volume
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_12h_aligned[i]
        vol_conf = vol_ratio[i] > 2.0
        
        if position == 0:
            # Bearish fractal + volume + price below EMA = short
            if bearish_fractal_confirmed[i] and vol_conf and price < ema34:
                signals[i] = -0.25
                position = -1
            # Bullish fractal + volume + price above EMA = long
            elif bullish_fractal_confirmed[i] and vol_conf and price > ema34:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit on bullish fractal (potential bottom) or if price crosses below EMA
            if bullish_fractal_confirmed[i] or price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on bearish fractal (potential top) or if price crosses above EMA
            if bearish_fractal_confirmed[i] or price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Volume_EMA34"
timeframe = "4h"
leverage = 1.0