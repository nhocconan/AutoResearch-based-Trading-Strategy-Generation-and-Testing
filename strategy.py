#!/usr/bin/env python3
"""
1d_WilliamsFractal_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Williams Fractal breakouts with weekly trend filter and volume confirmation.
Williams Fractals identify swing highs/lows that act as support/resistance.
Breakouts above bearish fractals or below bullish fractals with volume and weekly trend
capture strong moves. Works in bull/bear via weekly trend filter. Target: 30-100 trades over 4 years.
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
    
    # Volume confirmation: volume > 2.0x 20-period median (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    # Load weekly data for HTF trend filter (EMA34) and Williams Fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Williams Fractals on weekly data (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period volume median, 34-period EMA, and fractal lookback)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_median[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: close above bearish fractal (resistance broken) + volume confirm + bullish weekly trend
        if close[i] > bearish_fractal_aligned[i] and volume_confirm[i] and close[i] > ema34_1w_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: close below bullish fractal (support broken) + volume confirm + bearish weekly trend
        elif close[i] < bullish_fractal_aligned[i] and volume_confirm[i] and close[i] < ema34_1w_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: long exits when price touches bullish fractal (support), short exits when price touches bearish fractal (resistance)
        elif position == 1 and close[i] <= bullish_fractal_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] >= bearish_fractal_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0