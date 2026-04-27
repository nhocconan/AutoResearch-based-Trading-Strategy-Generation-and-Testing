#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend
Hypothesis: Williams fractal breakouts with daily trend filter capture momentum in both bull and bear markets.
Fractals require confirmation (2-bar delay) to avoid false breaks. 1d EMA34 filters trend direction.
Volume spikes confirm breakout strength. Target: 20-40 trades/year on 6h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams fractals (need 2-bar confirmation after center bar)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: volume > 2.0 * 20-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, fractals, volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]):
            signals[i] = 0.0
            continue
        
        ema_val = ema_34_aligned[i]
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        vol_confirm_val = vol_confirm[i]
        
        if position == 0:
            # Long: bullish fractal break above prior high, uptrend (price > EMA34), volume confirmation
            if bull_fract and close[i] > ema_val and vol_confirm_val:
                signals[i] = size
                position = 1
            # Short: bearish fractal break below prior low, downtrend (price < EMA34), volume confirmation
            elif bear_fract and close[i] < ema_val and vol_confirm_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price closes below EMA34 or opposing fractal appears
            if close[i] < ema_val or bear_fract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above EMA34 or opposing fractal appears
            if close[i] > ema_val or bull_fract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0