# 6h_WilliamsFractal_Breakout_1dTrend_Volume
# Hypothesis: Williams Fractals on 1D identify key reversal points; combining with 1D trend and volume breakout on 6H captures momentum after consolidation in both bull and bear markets. Fractals require 2-bar confirmation to avoid look-ahead.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Williams Fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Williams Fractals: bearish (high) and bullish (low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    n_1d = len(high_1d)
    
    bearish = np.zeros(n_1d, dtype=bool)
    bullish = np.zeros(n_1d, dtype=bool)
    
    # Williams Fractal: point is fractal if it's the highest/lowest of 5 bars (2 left, 2 right)
    for i in range(2, n_1d - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish[i] = True
    
    # 1d EMA50 for trend
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h: fractals need 2-bar confirmation (additional_delay_bars=2)
    bearish_fractal = bearish.astype(float)  # 1.0 where bearish fractal, 0 otherwise
    bullish_fractal = bullish.astype(float)
    
    bearish_6h = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_6h = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: 6h volume > 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_6h[i]) or np.isnan(bullish_6h[i]) or 
            np.isnan(ema50_1d_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition
        vol_ok = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: bullish fractal break above EMA50 with volume
            if bullish_6h[i] > 0 and close[i] > ema50_1d_6h[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal break below EMA50 with volume
            elif bearish_6h[i] > 0 and close[i] < ema50_1d_6h[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below EMA50 or bearish fractal
            if close[i] < ema50_1d_6h[i] or bearish_6h[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above EMA50 or bullish fractal
            if close[i] > ema50_1d_6h[i] or bullish_6h[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals