#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1d EMA50 trend filter and volume confirmation.
- Uses 1d Williams Fractals (bearish = short signal, bullish = long signal) for reversal signals
- Long entry: bullish fractal + volume > 1.5x 20-period avg + close > 1d EMA50 (uptrend)
- Short entry: bearish fractal + volume > 1.5x 20-period avg + close < 1d EMA50 (downtrend)
- Exit: opposite fractal signal or close crosses 1d EMA50
- Williams Fractals provide high-probability reversal points that work in ranging and trending markets
- 1d EMA50 ensures alignment with medium-term trend to avoid counter-trend trades
- Volume confirmation reduces false breakouts in low-participation moves
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe
- Williams Fractals are lagging indicators requiring 2-bar confirmation, properly handled with align_htf_to_ltf
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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Williams Fractals and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on 1d data
    # Bearish fractal: high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2] and low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n+2]
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    
    # Align Williams Fractals to 12h timeframe with 2-bar additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    ema_50_1d = pd.Series(close_1d := df_1d['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: bullish fractal + volume spike + close > 1d EMA50 (uptrend)
            if bullish_fractal_aligned[i] and volume_spike and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish fractal + volume spike + close < 1d EMA50 (downtrend)
            elif bearish_fractal_aligned[i] and volume_spike and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish fractal signal or close crosses below 1d EMA50
            if bearish_fractal_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish fractal signal or close crosses above 1d EMA50
            if bullish_fractal_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0