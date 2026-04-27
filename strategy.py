#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation.
- Williams Fractal identifies local swing points for swing trading in range/breakout markets
- 1d EMA34 trend filter ensures alignment with higher timeframe trend
- Volume spike confirms breakout strength
- Works in both bull/bear via breakout logic (long on upward fractal break, short on downward)
- Target: 15-30 trades/year to minimize fee drag
- Uses discrete position sizing (0.25) to minimize churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals (requires 5-point pattern)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Need 2-bar confirmation for fractals (per rule 2b)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate EMA34 on daily close for trend filter
    ema_34 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34[i] = (close_1d[i] * 2 + ema_34[i-1] * 33) / 35
    
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike: current volume > 2.0 * 20-period average (higher threshold for fewer trades)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(40, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_confirmed[i]) or 
            np.isnan(bullish_fractal_confirmed[i]) or
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: bullish fractal breakout + price above EMA34 + volume spike
            if (bullish_fractal_confirmed[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish fractal breakdown + price below EMA34 + volume spike
            elif (bearish_fractal_confirmed[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: bearish fractal breakdown OR price below EMA34
            if (bearish_fractal_confirmed[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish fractal breakout OR price above EMA34
            if (bullish_fractal_confirmed[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_EMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0