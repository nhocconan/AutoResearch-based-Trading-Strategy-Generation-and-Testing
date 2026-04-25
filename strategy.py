#!/usr/bin/env python3
"""
1d_WilliamsFractal_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Williams fractal breakouts (bullish/bearish) with 1-week EMA trend filter and volume confirmation (>1.8x 20-bar avg). Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear by following weekly trend. Williams fractals require 2-bar confirmation delay (additional_delay_bars=2). Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    # 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Williams fractals on 1d (requires 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need fractals (2-bar delay), volume MA (20), 1w EMA (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above bullish fractal in 1w uptrend with volume spike
            long_signal = (curr_close > bullish_fractal_aligned[i]) and \
                         (ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]) and \
                         volume_spike[i]
            # Short: price breaks below bearish fractal in 1w downtrend with volume spike
            short_signal = (curr_close < bearish_fractal_aligned[i]) and \
                          (ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]) and \
                          volume_spike[i]
            
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
            # Exit: price breaks below bearish fractal OR trend turns down
            if (curr_close < bearish_fractal_aligned[i]) or \
               (ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above bullish fractal OR trend turns up
            if (curr_close > bullish_fractal_aligned[i]) or \
               (ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0