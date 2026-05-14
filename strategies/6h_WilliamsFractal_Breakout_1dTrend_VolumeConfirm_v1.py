#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: On 6h timeframe, Williams fractal breaks (beyond prior swing high/low) with 1d EMA34 trend filter and volume confirmation capture strong momentum moves. Williams fractals require 2-bar confirmation on HTF, reducing false breaks. Volume spike confirms participation. 1d EMA34 ensures alignment with longer-term trend. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Load 1d data ONCE before loop for HTF trend filter (EMA) and fractals
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams fractals on 1d (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar extra delay for fractal confirmation (needs 2 future 1d bars)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume spike detection on 6h (volume > 1.8x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
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
        
        # 1d trend filter (EMA34)
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Long logic: price breaks above bullish fractal with volume spike + uptrend
        if close[i] > bullish_aligned[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below bearish fractal with volume spike + downtrend
        elif close[i] < bearish_aligned[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price reaches opposite fractal or trend reversal
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

name = "6h_WilliamsFractal_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0