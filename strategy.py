#!/usr/bin/env python3
"""
6h_Williams_Fractal_Donchian_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Williams Fractal (1d) identifies swing points; 6h Donchian(20) breakout in direction of 1d EMA50 trend with volume confirmation (>2x 20-bar avg). Fractals require 2-bar confirmation delay. Discrete sizing (0.25) limits trades to ~20-40/year. Works in bull/bear via trend filter: only long when price > EMA50, short when price < EMA50. Volume spike confirms momentum. Avoids saturated Camarilla/Donchian families by adding fractal structure for swing-based breakouts.
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
    
    # Get 1d data for HTF trend and fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams Fractals on 1d (requires 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Bearish fractal: high[2] is highest of [0..4]; needs 2 future bars to confirm
    # Bullish fractal: low[2] is lowest of [0..4]; needs 2 future bars to confirm
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 6h Donchian(20) channels
    donchian_h20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-bar average volume for confirmation on 6h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian, volume MA20, EMA50
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(donchian_h20[i]) or
            np.isnan(donchian_l20[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 2.0x 20-bar average
            volume_confirm = volume[i] > 2.0 * vol_ma20[i]
            
            # Long: price breaks above Donchian H20 in uptrend (price > EMA50) with volume spike
            #        and bullish fractal confirmed (support level intact)
            long_signal = (close[i] > donchian_h20[i]) and (close[i] > ema50_1d_aligned[i]) and volume_confirm and bullish_fractal_aligned[i]
            
            # Short: price breaks below Donchian L20 in downtrend (price < EMA50) with volume spike
            #        and bearish fractal confirmed (resistance level intact)
            short_signal = (close[i] < donchian_l20[i]) and (close[i] < ema50_1d_aligned[i]) and volume_confirm and bearish_fractal_aligned[i]
            
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
            # Exit when price moves back below Donchian L20 (breakdown) or trend turns bearish
            exit_signal = (close[i] < donchian_l20[i]) or (close[i] < ema50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian H20 (breakout) or trend turns bullish
            exit_signal = (close[i] > donchian_h20[i]) or (close[i] > ema50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Williams_Fractal_Donchian_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0