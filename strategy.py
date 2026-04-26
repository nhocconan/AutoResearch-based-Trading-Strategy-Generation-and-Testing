#!/usr/bin/env python3
"""
1d_WilliamsFractal_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Williams fractal breaks on 1d with 1w EMA50 trend filter and volume spike capture major swing continuations. Fractals identify key swing highs/lows; breaks indicate momentum acceleration. Volume spike confirms validity. 1w EMA50 ensures alignment with weekly trend. Target: 30-80 total trades over 4 years (7-20/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Load 1d data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams fractals on 1d (requires 2-bar confirmation after center bar)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Align with 2 extra bars delay for fractal confirmation
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume spike detection on 1d (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or
            np.isnan(bullish_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (EMA50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Long logic: price breaks above bullish fractal with volume spike + in uptrend
        if close[i] > bullish_aligned[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below bearish fractal with volume spike + in downtrend
        elif close[i] < bearish_aligned[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite fractal level or trend weakens
        elif position == 1 and (close[i] < bearish_aligned[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > bullish_aligned[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0