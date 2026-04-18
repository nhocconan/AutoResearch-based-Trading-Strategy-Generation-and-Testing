#!/usr/bin/env python3
"""
4h_WilliamsFractal_Breakout_VolumeConfirm_V1
Hypothesis: Williams Fractal breakouts on 4h with volume confirmation and 1d trend filter.
Go long when price breaks above recent bearish fractal high AND 1d EMA50 is rising, 
short when price breaks below recent bullish fractal low AND 1d EMA50 is falling.
Requires volume > 1.3x 20-period average. Target: 20-40 trades/year by using fractal 
structure for support/resistance breaks. Works in trending markets via breakouts and 
in ranging markets via mean reversion at fractal levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for fractals
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Williams Fractals: bearish (high) and bullish (low)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n_4h = len(high_4h)
    bearish_fractal = np.zeros(n_4h)
    bullish_fractal = np.zeros(n_4h)
    
    for i in range(2, n_4h - 2):
        if (high_4h[i-2] < high_4h[i-1] and high_4h[i] < high_4h[i-1] and
            high_4h[i-3] < high_4h[i-1] and high_4h[i+1] < high_4h[i-1]):
            bearish_fractal[i-1] = high_4h[i-1]
        if (low_4h[i-2] > low_4h[i-1] and low_4h[i] > low_4h[i-1] and
            low_4h[i-3] > low_4h[i-1] and low_4h[i+1] > low_4h[i-1]):
            bullish_fractal[i-1] = low_4h[i-1]
    
    # Recent fractal levels: carry forward the last fractal
    bearish_level = np.full(n_4h, np.nan)
    bullish_level = np.full(n_4h, np.nan)
    last_bear = np.nan
    last_bull = np.nan
    for i in range(n_4h):
        if not np.isnan(bearish_fractal[i]):
            last_bear = bearish_fractal[i]
        if not np.isnan(bullish_fractal[i]):
            last_bull = bullish_fractal[i]
        bearish_level[i] = last_bear
        bullish_level[i] = last_bull
    
    # Align fractal levels to 4h timeframe (no extra delay needed as fractals are confirmed on same bar)
    bearish_level_aligned = align_htf_to_ltf(prices, df_4h, bearish_level)
    bullish_level_aligned = align_htf_to_ltf(prices, df_4h, bullish_level)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50
    ema_period = 50
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA50 to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 2) + 1  # fractals need 2 bars lookback/forward
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bearish_level_aligned[i]) or np.isnan(bullish_level_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above recent bearish fractal resistance AND 1d EMA50 rising
            if close[i] > bearish_level_aligned[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below recent bullish fractal support AND 1d EMA50 falling
            elif close[i] < bullish_level_aligned[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below recent bullish fractal support OR 1d EMA50 turns down
            if close[i] < bullish_level_aligned[i] or ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above recent bearish fractal resistance OR 1d EMA50 turns up
            if close[i] > bearish_level_aligned[i] or ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsFractal_Breakout_VolumeConfirm_V1"
timeframe = "4h"
leverage = 1.0