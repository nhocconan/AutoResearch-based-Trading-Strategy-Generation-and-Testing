#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_12hTrend_VolumeSpike
Hypothesis: Williams fractal breaks on 6h timeframe with 12h EMA50 trend filter and volume confirmation works in both bull and bear markets. Fractals provide structure-based support/resistance that adapts to volatility, while EMA50 on 12h captures medium-term trend. Volume spike confirms institutional participation. Targets 15-25 trades/year for low fee drag.
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
    
    # Get 12h HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Williams Fractals on 6L time (need 5 bars: 2 left, center, 2 right)
    # Bearish fractal: high[n] is highest among [n-2, n-1, n, n+1, n+2]
    # Bullish fractal: low[n] is lowest among [n-2, n-1, n, n+1, n+2]
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    
    # Need 2 extra 12h bars for fractal confirmation (center + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0 * 50-period average (strict for 6h)
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 50 for volume avg/EMA, plus fractal lookback
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Conservative position size for 6h
        
        if position == 0:
            # Flat - look for fractal breakout with trend and volume confirmation
            # Long: price breaks above bearish fractal (resistance) + 12h EMA50 uptrend + volume spike
            long_entry = (close_val > bearish_fractal_aligned[i]) and \
                       (ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]) and \
                       volume_spike[i]
            # Short: price breaks below bullish fractal (support) + 12h EMA50 downtrend + volume spike
            short_entry = (close_val < bullish_fractal_aligned[i]) and \
                        (ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price retests the broken fractal level (now support) or trend fails
            if (close_val < bearish_fractal_aligned[i]) or (ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price retests the broken fractal level (now resistance) or trend fails
            if (close_val > bullish_fractal_aligned[i]) or (ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0