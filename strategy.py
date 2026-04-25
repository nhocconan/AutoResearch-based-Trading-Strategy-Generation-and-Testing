#!/usr/bin/env python3
"""
6h_WilliamsFractal_1dTrend_VolumeBreakout
Hypothesis: Williams fractals on 1d confirm swing points; 6h breaks above/below recent fractal levels with volume spike and 1d EMA50 trend filter capture institutional participation in both bull and bear regimes. Fractals require 2-bar confirmation to avoid look-ahead.
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
    
    # 1d data for Williams fractals, EMA trend, and ATR (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams fractals on 1d high/low with 2-bar confirmation delay
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Additional delay of 2 bars for confirmation (fractal forms at bar n, confirmed at bar n+2)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 6h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 1d EMA (50) + volume MA (20) + fractal formation (5 bars) + alignment delay
    start_idx = max(50, 20, 5) + 5  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above recent bullish fractal (resistance turned support) with volume spike and 1d uptrend
            long_breakout = (curr_close > bullish_aligned[i]) and vol_spike[i] and (curr_close > ema_aligned[i])
            # Short: price breaks below recent bearish fractal (support turned resistance) with volume spike and 1d downtrend
            short_breakout = (curr_close < bearish_aligned[i]) and vol_spike[i] and (curr_close < ema_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below recent bullish fractal or trend turns down
            if (curr_close < bullish_aligned[i]) or (curr_close < ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above recent bearish fractal or trend turns up
            if (curr_close > bearish_aligned[i]) or (curr_close > ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_1dTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0