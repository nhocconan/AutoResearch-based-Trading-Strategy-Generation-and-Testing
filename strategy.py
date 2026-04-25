#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrendFilter_v1
Hypothesis: Trade Williams Fractal breakouts on 6h with 1d EMA34 trend filter.
Williams Fractals identify potential reversal points; breakouts above/below recent fractals
with 1d trend alignment capture momentum in both bull and bear markets.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Compute Williams Fractals on 1d data (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar delay for fractal confirmation (needs 2 future 1d bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and fractals (5 bars for pattern + 2 delay)
    start_idx = max(34, 5 + 2)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above bullish fractal + 1d uptrend
            long_setup = (close[i] > bullish_fractal_aligned[i]) and htf_1d_bullish
            
            # Short setup: price breaks below bearish fractal + 1d downtrend
            short_setup = (close[i] < bearish_fractal_aligned[i]) and htf_1d_bearish
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches bearish fractal (stop) OR 1d trend turns bearish
            if (close[i] <= bearish_fractal_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches bullish fractal (stop) OR 1d trend turns bullish
            if (close[i] >= bullish_fractal_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0