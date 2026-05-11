#!/usr/bin/env python3
name = "6h_WilliamsFractal_Swing_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals on 12h (need 2-bar confirmation)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_12h['high'].values,
        df_12h['low'].values,
    )
    # Need 2 extra 12h bars for confirmation (Williams fractal rule)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bullish_fractal, additional_delay_bars=2
    )
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter (20-bar average)
    volume_ma20 = pd.Series(volume).rolling(
        window=20, min_periods=20
    ).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # EMA50 and volume MA ready
    
    for i in range(start_idx, n):
        if np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        vol_ok = volume[i] > 1.5 * volume_ma20[i]
        
        if position == 0:
            # Long: bullish fractal + price above EMA50 + volume
            if bullish_fractal_aligned[i] and close[i] > ema_50_12h_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal + price below EMA50 + volume
            elif bearish_fractal_aligned[i] and close[i] < ema_50_12h_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish fractal or price below EMA50
            if bearish_fractal_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish fractal or price above EMA50
            if bullish_fractal_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals