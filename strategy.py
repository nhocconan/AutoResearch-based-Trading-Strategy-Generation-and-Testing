#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend_HTFVolume_v1
Hypothesis: On 6h timeframe, Williams Fractals from 1d identify key swing points; breakouts above recent bullish fractal highs or below bearish fractal lows with 1d EMA trend filter and 1d volume confirmation capture strong moves in both bull and bear markets. Fractals require 2-bar confirmation, reducing false signals. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams Fractals on 1d (requires 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2 extra bars delay for fractal confirmation completion
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for EMA and fractals)
    start_idx = max(40, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
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
        
        # 1d volume confirmation: volume > 1.5x 20-period EMA
        volume_spike = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Fractal breakout conditions
        bullish_breakout = close[i] > bullish_fractal_aligned[i]
        bearish_breakout = close[i] < bearish_fractal_aligned[i]
        
        # Long logic: breakout above bullish fractal in uptrend with volume
        if uptrend and volume_spike and bullish_breakout:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below bearish fractal in downtrend with volume
        elif downtrend and volume_spike and bearish_breakout:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not downtrend:
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

name = "6h_WilliamsFractal_Breakout_1dTrend_HTFVolume_v1"
timeframe = "6h"
leverage = 1.0