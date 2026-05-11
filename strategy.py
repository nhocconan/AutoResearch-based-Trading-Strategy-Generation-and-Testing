#!/usr/bin/env python3
name = "12h_WilFractal_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Williams Fractals (requires 2 extra bars for confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Add 2-bar delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1d EMA trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish fractal + above 1d EMA + volume
            if bullish_fractal_aligned[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal + below 1d EMA + volume
            elif bearish_fractal_aligned[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish fractal or below 1d EMA
            if bearish_fractal_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish fractal or above 1d EMA
            if bullish_fractal_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals