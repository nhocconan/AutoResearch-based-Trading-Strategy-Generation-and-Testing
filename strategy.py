#!/usr/bin/env python3
name = "12h_Williams_Fractal_Trend_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: align_ltf_to_htf is not used; we use align_htf_to_ltf
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1W data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 1D data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1W EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Williams Fractals on 1D (requires 2-bar confirmation after center)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    n_1d = len(high_1d)
    
    bullish_fractal = np.zeros(n_1d, dtype=bool)
    bearish_fractal = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        # Bullish fractal: low[i] is lowest of i-2,i-1,i,i+1,i+2
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish_fractal[i] = True
        # Bearish fractal: high[i] is highest of i-2,i-1,i,i+1,i+2
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish_fractal[i] = True
    
    # Align fractals with 2-bar additional delay for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Price above/below 1W EMA34 for trend filter
        price_above_ema = close[i] > ema34_1w_aligned[i]
        price_below_ema = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: Bullish fractal confirmed, price above 1W EMA34, volume surge
            if bullish_fractal_aligned[i] > 0.5 and price_above_ema and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal confirmed, price below 1W EMA34, volume surge
            elif bearish_fractal_aligned[i] > 0.5 and price_below_ema and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish fractal OR price crosses below 1W EMA34
            if bearish_fractal_aligned[i] > 0.5 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish fractal OR price crosses above 1W EMA34
            if bullish_fractal_aligned[i] > 0.5 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals