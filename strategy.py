#!/usr/bin/env python3
"""
12h Williams Fractal Breakout with Volume Confirmation and EMA Trend Filter
Uses 1-day Williams fractals for breakout entries, confirmed by 1-day volume spikes
and 1-day EMA trend direction. Targets 50-150 total trades over 4 years (12-37/year)
to minimize fee drift while capturing breakout momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals, volume, and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams fractals (need 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Apply 2-bar additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1-day volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate 1-day EMA(50) for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Bullish fractal breakout + volume spike + uptrend
        bullish_breakout = high[i] > bullish_fractal_aligned[i]
        # Entry conditions: Bearish fractal breakdown + volume spike + downtrend
        bearish_breakdown = low[i] < bearish_fractal_aligned[i]
        vol_confirm = vol_spike_aligned[i] > 0.5
        uptrend = close_1d[-1] > ema_50_aligned[i] if len(close_1d) > 0 else False  # Simplified: use current close vs EMA
        downtrend = close_1d[-1] < ema_50_aligned[i] if len(close_1d) > 0 else False
        
        long_entry = bullish_breakout and vol_confirm and uptrend
        short_entry = bearish_breakdown and vol_confirm and downtrend
        
        # Exit when price returns to the fractal level
        exit_long = position == 1 and low[i] <= bullish_fractal_aligned[i]
        exit_short = position == -1 and high[i] >= bearish_fractal_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_williams_fractal_breakout"
timeframe = "12h"
leverage = 1.0