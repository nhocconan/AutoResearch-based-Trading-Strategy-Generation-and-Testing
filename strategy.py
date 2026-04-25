#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend_Filter_v2
Hypothesis: Williams fractal breaks on 6h with 1d EMA50 trend filter and volume confirmation.
Only trade when 6h volume > 1.5x 20-period average to avoid false breakouts in low volume.
Fractals require 2-bar confirmation. Discrete sizing 0.25 for low trade frequency (~15-25/year).
Designed to work in both bull and bear markets by aligning with daily trend and filtering weak breakouts.
"""

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
    
    # Get 1d data for HTF trend filter and Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams fractals on 1d data (requires 2-bar confirmation)
    bearish_fractal_1d, bullish_fractal_1d = compute_williams_fractals(high_1d, low_1d)
    # Fractals need 2 extra bars for confirmation beyond the standard HTF delay
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_1d, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_1d, additional_delay_bars=2)
    
    # Align HTF EMA50 to 6h timeframe (standard 1-bar delay for EMA)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate 20-period average volume on 6h for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for fractal breakout signals with trend filter and volume confirmation
            # Long: price breaks above bullish fractal in uptrend (close > EMA50) + volume
            # Short: price breaks below bearish fractal in downtrend (close < EMA50) + volume
            long_signal = (close[i] > bullish_fractal_aligned[i]) and (close[i] > ema50_aligned[i]) and volume_confirm
            short_signal = (close[i] < bearish_fractal_aligned[i]) and (close[i] < ema50_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below EMA50 (trend reversal)
            exit_signal = close[i] < ema50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above EMA50 (trend reversal)
            exit_signal = close[i] > ema50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dTrend_Filter_v2"
timeframe = "6h"
leverage = 1.0