#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WilliamsFractal_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === Williams Fractals on daily chart ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    bearish = np.zeros(len(high_1d), dtype=bool)
    bullish = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i+2]):
            bearish[i] = True
            
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i+2]):
            bullish[i] = True
    
    # Convert to levels: bearish fractal = resistance level, bullish = support
    fractal_resist = np.where(bearish, high_1d, np.nan)
    fractal_support = np.where(bullish, low_1d, np.nan)
    
    # Forward fill to get the most recent fractal level
    resist_series = pd.Series(fractal_resist)
    resist_ffill = resist_series.ffill().values
    support_series = pd.Series(fractal_support)
    support_ffill = support_series.ffill().values
    
    # Align to 6h timeframe with 2-bar delay for confirmation
    resist_aligned = align_htf_to_ltf(prices, df_1d, resist_ffill, additional_delay_bars=2)
    support_aligned = align_htf_to_ltf(prices, df_1d, support_ffill, additional_delay_bars=2)
    
    # === 6h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma20 > 0, volume / vol_ma20, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        close_val = prices['close'].iloc[i]
        resist_val = resist_aligned[i]
        support_val = support_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(resist_val) or np.isnan(support_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above recent bearish fractal (resistance) with volume
            if close_val > resist_val and vol_ratio_val > 2.0:
                signals[i] = 0.25
                position = 1
            # Short: Break below recent bullish fractal (support) with volume
            elif close_val < support_val and vol_ratio_val > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below support fractal OR volume dries up
            if close_val < support_val or vol_ratio_val < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above resistance fractal OR volume dries up
            if close_val > resist_val or vol_ratio_val < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals