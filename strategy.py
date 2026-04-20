#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WilliamsFractal_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # === 1d: Williams Fractals (5-point: 2 left, 2 right) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    # Need at least 5 points: i-2, i-1, i, i+1, i+2
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: high[i] is highest of the 5
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal: low[i] is lowest of the 5
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams fractals need 2 extra bars for confirmation (after the center bar)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(bear_fractal) or np.isnan(bull_fractal) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above bullish fractal (support) with volume
            if (close_val > bull_fractal and vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bearish fractal (resistance) with volume
            elif (close_val < bear_fractal and vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below bullish fractal or low volume
            if (close_val < bull_fractal or vol_ratio_val < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above bearish fractal or low volume
            if (close_val > bear_fractal or vol_ratio_val < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals