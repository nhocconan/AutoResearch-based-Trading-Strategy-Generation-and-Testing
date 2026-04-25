#!/usr/bin/env python3
"""
6h Williams Fractal Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Williams fractals identify significant swing points where price respects structure.
Bullish fractal breakout (price > recent bearish fractal high) with 1d EMA50 uptrend and volume spike signals continuation.
Bearish fractal breakdown (price < recent bullish fractal low) with 1d EMA50 downtrend and volume spike signals continuation.
Uses 6h timeframe for lower frequency (target 12-37 trades/year) and 1d EMA50 as HTF trend filter.
Discrete sizing (0.0, ±0.25) minimizes fee churn. Works in both bull (trend continuation) and bear (fades at structure).
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
    
    # Get 1d data for EMA trend filter and fractals (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams fractals on 1d (need 2-bar confirmation after center)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Bearish fractal: extra 2 bars for confirmation (total 3-bar pattern: left, center, right)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    # Bullish fractal: extra 2 bars for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current volume > 1.5 * 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = 50 + 20 + 2  # 1d EMA50 + vol MA + fractal confirmation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_1d_aligned[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above recent bearish fractal (resistance) AND volume spike AND price > 1d EMA50 (uptrend)
            long_entry = (curr_close > bear_fractal) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below recent bullish fractal (support) AND volume spike AND price < 1d EMA50 (downtrend)
            short_entry = (curr_close < bull_fractal) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below bullish fractal (support break) OR trend change (price < EMA)
            if (curr_close < bull_fractal) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above bearish fractal (resistance break) OR trend change (price > EMA)
            if (curr_close > bear_fractal) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0