#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses 6h timeframe (primary) and 1d HTF for EMA34 trend alignment.
- Williams Fractals identify swing highs/lows from 1d data (requires 2-bar confirmation).
- Breakout logic: long when price closes above latest bullish fractal with volume spike and uptrend,
                  short when price closes below latest bearish fractal with volume spike and downtrend.
- Trend filter: only long when 6h close > 1d EMA34, only short when 6h close < 1d EMA34.
- Volume confirmation: current 6h volume > 2.0 * 20-period 6h volume MA (strict to reduce trades).
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull/bear: trend filter avoids counter-trend trades, fractal breakouts capture momentum in all regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Fractals from 1d data (requires 2 extra bars for confirmation)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Align fractals to 6h timeframe with additional 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (strict)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Trend filter: 6h close vs 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34)  # Need 1d EMA34 and sufficient volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above bullish fractal AND uptrend AND volume spike
            if close[i] > bullish_fractal_aligned[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below bearish fractal AND downtrend AND volume spike
            elif close[i] < bearish_fractal_aligned[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to bearish fractal level or reverse signal
            if close[i] <= bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to bullish fractal level or reverse signal
            if close[i] >= bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0