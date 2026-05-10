#!/usr/bin/env python3
"""
6h_FractalBreakout_1dTrend_Volume
Hypothesis: Use daily Williams fractals to identify key support/resistance levels.
In trending markets (determined by 1d EMA34), price breaks through recent fractal levels
with volume acceleration, signaling continuation. Fractals require 2-bar confirmation,
reducing false breaks. Works in both bull (upward breakouts) and bear (downward breakdowns).
Targets 50-120 trades over 4 years (12-30/year) to minimize fee drag.
"""

name = "6h_FractalBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Daily Williams fractals (requires 2-bar confirmation)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Add 2-bar delay for confirmation (fractal forms after 2nd bar closes)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 2)  # Need EMA34 and fractal confirmation
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or \
           np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 1d volume (scaled to 6h)
        vol_6h_approx = vol_sma20_1d_aligned[i] / 4.0  # 4x 6h periods in 1d
        volume_confirm = volume[i] > 1.5 * vol_6h_approx
        
        if position == 0:
            # Long: Break above recent bullish fractal (resistance) in uptrend with volume
            if bullish_fractal_aligned[i] > 0 and close[i] > bullish_fractal_aligned[i] and \
               close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below recent bearish fractal (support) in downtrend with volume
            elif bearish_fractal_aligned[i] > 0 and close[i] < bearish_fractal_aligned[i] and \
                 close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price falls below fractal level or trend reversal
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above fractal level or trend reversal
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals