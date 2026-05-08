#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_Breakout_1dTrend"
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
    
    # Get 1d data for Williams Fractals and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals: 5-bar window (2 left, 2 right)
    # Bearish fractal: high[n-2] < high[n] and high[n-1] < high[n] and high[n+1] < high[n] and high[n+2] < high[n]
    # Bullish fractal: low[n-2] > low[n] and low[n-1] > low[n] and low[n+1] > low[n] and low[n+2] > low[n]
    n1d = len(high_1d)
    bearish_fractal = np.zeros(n1d, dtype=bool)
    bullish_fractal = np.zeros(n1d, dtype=bool)
    
    for i in range(2, n1d - 2):
        if (high_1d[i-2] < high_1d[i] and high_1d[i-1] < high_1d[i] and 
            high_1d[i+1] < high_1d[i] and high_1d[i+2] < high_1d[i]):
            bearish_fractal[i] = True
        if (low_1d[i-2] > low_1d[i] and low_1d[i-1] > low_1d[i] and 
            low_1d[i+1] > low_1d[i] and low_1d[i+2] > low_1d[i]):
            bullish_fractal[i] = True
    
    # Williams Fractals need 2-bar confirmation delay after the center bar
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish fractal breakout, price above 1d EMA34, volume spike
            long_cond = bullish_fractal_aligned[i] > 0 and \
                       close[i] > ema34_1d_aligned[i] and \
                       volume_spike[i]
            
            # Short: Bearish fractal breakdown, price below 1d EMA34, volume spike
            short_cond = bearish_fractal_aligned[i] > 0 and \
                        close[i] < ema34_1d_aligned[i] and \
                        volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below 1d EMA34 OR bearish fractal appears
            if close[i] < ema34_1d_aligned[i] or bearish_fractal_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above 1d EMA34 OR bullish fractal appears
            if close[i] > ema34_1d_aligned[i] or bullish_fractal_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals