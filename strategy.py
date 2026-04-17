#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w Williams Fractal breakout and volume confirmation.
Trade breakouts of weekly Williams Fractal levels (bullish/bearish) with volume spike (>1.5x 20-period average).
Use 1w ADX > 20 to filter for trending markets and avoid ranging whipsaws.
In trending markets: buy breakouts above weekly bearish fractal (resistance), sell breakdowns below weekly bullish fractal (support).
Position sizing: 0.25 for entries, 0 for exits.
Target: 30-100 total trades over 4 years (7-25/year).
Williams Fractals from 1w provide significant structural levels that work in both bull and bear markets by identifying genuine swing points.
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
    
    # Get 1w data for Williams Fractals and ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w Williams Fractals (requires 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    # Additional 2-bar delay needed for fractal confirmation (see Rule 2b)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1w ADX (14)
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1d
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    plus_di_aligned = align_htf_to_ltf(prices, df_1w, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1w, minus_di)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from ADX components
        uptrend = plus_di_aligned[i] > minus_di_aligned[i]
        downtrend = plus_di_aligned[i] < minus_di_aligned[i]
        strong_trend = adx_aligned[i] > 20
        
        if position == 0:
            # Long: price breaks above bearish fractal (resistance), volume spike, strong trend
            if (close[i] > bearish_fractal_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                strong_trend and uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal (support), volume spike, strong trend
            elif (close[i] < bullish_fractal_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                  strong_trend and downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below bullish fractal or trend weakens
            if close[i] < bullish_fractal_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above bearish fractal or trend weakens
            if close[i] > bearish_fractal_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wWilliamsFractal_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0