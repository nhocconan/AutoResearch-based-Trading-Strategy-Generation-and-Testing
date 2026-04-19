#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WilliamsFractal_Reversal_Signal_v1"
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
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals (need 2 bars on each side)
    # Bearish fractal: high[n] > high[n-1] and high[n] > high[n-2] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-1] and low[n] < low[n-2] and low[n] < low[n+1] and low[n] < low[n+2]
    n_1d = len(high_1d)
    bearish_fractal = np.zeros(n_1d, dtype=bool)
    bullish_fractal = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Convert to float arrays for alignment (1.0 where fractal exists, 0 otherwise)
    bearish_fractal_float = bearish_fractal.astype(float)
    bullish_fractal_float = bullish_fractal.astype(float)
    
    # Williams fractal needs 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal_float, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal_float, additional_delay_bars=2
    )
    
    # EMA filter on 1d close for trend bias
    ema_period = 50
    ema_1d = pd.Series(df_1d['close'].values).ewm(
        span=ema_period, adjust=False, min_periods=ema_period
    ).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema = ema_1d_aligned[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Bullish fractal confirmed + price above EMA + volume spike
            if bullish_fractal_aligned[i] == 1.0 and price > ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal confirmed + price below EMA + volume spike
            elif bearish_fractal_aligned[i] == 1.0 and price < ema and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bearish fractal appears (potential top) OR price crosses below EMA
            if bearish_fractal_aligned[i] == 1.0 or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bullish fractal appears (potential bottom) OR price crosses above EMA
            if bullish_fractal_aligned[i] == 1.0 or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals