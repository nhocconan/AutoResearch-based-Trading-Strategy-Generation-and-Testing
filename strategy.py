#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_Breakout_1dTrend_VolumeFilter"
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
    
    # Get 1d data for Williams fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Williams fractal on 1d (requires 5 bars: 2 left, center, 2 right)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bearish fractal: high[n] is highest of [n-2, n-1, n, n+1, n+2]
    bearish = np.zeros(len(high_1d), dtype=bool)
    bullish = np.zeros(len(low_1d), dtype=bool)
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish[i] = True
    
    # Convert to price levels (use the high/low of the fractal bar)
    bearish_level = np.where(bearish, high_1d, np.nan)
    bullish_level = np.where(bullish, low_1d, np.nan)
    
    # 1d EMA50 for trend
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h (Williams fractals need 2-bar confirmation delay)
    bearish_6h = align_htf_to_ltf(prices, df_1d, bearish_level, additional_delay_bars=2)
    bullish_6h = align_htf_to_ltf(prices, df_1d, bullish_level, additional_delay_bars=2)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_1d_6h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_6h[i]) or np.isnan(bullish_6h[i]) or 
            np.isnan(ema50_1d_6h[i]) or np.isnan(vol_avg_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_1d_6h[i]
        bear_fractal = bearish_6h[i]
        bull_fractal = bullish_6h[i]
        vol_avg = vol_avg_1d_6h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        if position == 0:
            # Long: break above bullish fractal with volume and above 1d EMA50
            if close[i] > bull_fractal and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below bearish fractal with volume and below 1d EMA50
            elif close[i] < bear_fractal and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below bullish fractal or trend reversal
            if close[i] < bull_fractal or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above bearish fractal or trend reversal
            if close[i] > bear_fractal or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals