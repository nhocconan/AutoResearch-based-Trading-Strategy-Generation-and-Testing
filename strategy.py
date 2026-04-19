#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_1d_Trend_Signal_v1"
timeframe = "6h"
leverage = 1.0

def calculate_williams_fractals(high, low, n=2):
    """Calculate Williams Fractals: bearish (high) and bullish (low)"""
    high = np.asarray(high)
    low = np.asarray(low)
    bearish = np.zeros(len(high), dtype=bool)
    bullish = np.zeros(len(low), dtype=bool)
    
    for i in range(n, len(high) - n):
        # Bearish fractal: high[i] is highest in window [i-n, i+n]
        if high[i] == np.max(high[i-n:i+n+1]):
            bearish[i] = True
        # Bullish fractal: low[i] is lowest in window [i-n, i+n]
        if low[i] == np.min(low[i-n:i+n+1]):
            bullish[i] = True
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low for Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals on daily timeframe
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d, n=2)
    
    # Convert to float for alignment (True=1.0, False=0.0)
    bearish_fractal_float = bearish_fractal.astype(float)
    bullish_fractal_float = bullish_fractal.astype(float)
    
    # Align fractals to 6h timeframe with extra delay for confirmation
    # Williams fractals need 2 extra daily bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_float, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_float, additional_delay_bars=2)
    
    # 60-period EMA on 6h for trend filter
    close_series = pd.Series(close)
    ema_60 = close_series.ewm(span=60, adjust=False, min_periods=60).values
    
    # Volume confirmation: current volume > 1.5x 20-period average (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_60[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema = ema_60[i]
        bearish = bearish_fractal_aligned[i] > 0.5
        bullish = bullish_fractal_aligned[i] > 0.5
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: bullish fractal + price above EMA + volume
            if bullish and price > ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal + price below EMA + volume
            elif bearish and price < ema and volume_confirmed:
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