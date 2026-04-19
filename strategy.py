#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_1d_Trend_Signal_v2"
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
    
    # Get daily data for Williams fractal calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams fractal on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams fractal: bearish fractal (sell signal) = high > high[-2] and high > high[-1] and high > high[+1] and high > high[+2]
    # bullish fractal (buy signal) = low < low[-2] and low < low[-1] and low < low[+1] and low < low[+2]
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Align daily fractal signals to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_float = bearish_fractal.astype(float)
    bullish_fractal_float = bullish_fractal.astype(float)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_float, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_float, additional_delay_bars=2)
    
    # Trend filter: 50-period EMA on 6h timeframe
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema = ema_50[i]
        bearish = bearish_aligned[i] > 0.5
        bullish = bullish_aligned[i] > 0.5
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: bullish fractal with volume and above EMA
            if bullish and volume_confirmed and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal with volume and below EMA
            elif bearish and volume_confirmed and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: bearish fractal or price below EMA
            if bearish or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: bullish fractal or price above EMA
            if bullish or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals