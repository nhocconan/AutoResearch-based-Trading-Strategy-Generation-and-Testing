#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation.
- Long: Close > bullish fractal AND price > 1w EMA50 AND volume > 1.8x 20-period avg
- Short: Close < bearish fractal AND price < 1w EMA50 AND volume > 1.8x 20-period avg
- Exit: Opposite fractal breakout OR price crosses 1w EMA50
- Uses 1w HTF for EMA50 and fractals (calculated from prior 1w bar)
- Designed for low trade frequency (7-25/year) to minimize fee drag on 1d timeframe
- Works in bull (buy breakouts above bullish fractal) and bear (sell breakdowns below bearish fractal)
- Williams fractals require 2-bar confirmation delay to avoid look-ahead
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
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Fractals from prior 1w bar (HTF = 1w)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    # Williams fractals need 2 extra 1w bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Fractal breakout signals (using current close vs prior levels)
        breakout_up = close[i] > bullish_fractal_aligned[i-1]  # Close above prior bullish fractal
        breakout_down = close[i] < bearish_fractal_aligned[i-1]  # Close below prior bearish fractal
        
        if position == 0:
            # Long: Bullish fractal breakout up AND price > 1w EMA50 AND volume confirmation
            if breakout_up and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal breakout down AND price < 1w EMA50 AND volume confirmation
            elif breakout_down and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish fractal breakout down OR price < 1w EMA50 (trend flip)
            if breakout_down or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish fractal breakout up OR price > 1w EMA50 (trend flip)
            if breakout_up or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Fractal_Breakout_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0