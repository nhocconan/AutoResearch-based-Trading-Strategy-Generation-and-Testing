#!/usr/bin/env python3
name = "6h_1d_WilliamsFractal_Pullback_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Fractals on daily (need 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Bearish fractal = potential resistance (sell signal)
    # Bullish fractal = potential support (buy signal)
    bearish_fractal_confirmed = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_confirmed = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Daily trend filter: EMA(50) on close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bearish_fractal_confirmed[i]) or 
            np.isnan(bullish_fractal_confirmed[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: pullback to bullish fractal support in uptrend with volume
            vol_condition = volume[i] > vol_ma_24[i] * 1.5
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] > bullish_fractal_confirmed[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: pullback to bearish fractal resistance in downtrend with volume
            elif close[i] < bearish_fractal_confirmed[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below bullish fractal or trend changes
            if close[i] < bullish_fractal_confirmed[i] or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above bearish fractal or trend changes
            if close[i] > bearish_fractal_confirmed[i] or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Williams Fractal pullback with daily trend and volume confirmation
# - Williams Fractals identify key support/resistance levels on daily chart
# - In uptrend: buy pullbacks to bullish fractal support with volume confirmation
# - In downtrend: sell pullbacks to bearish fractal resistance with volume confirmation
# - Fractals require 2-bar confirmation, reducing false signals
# - Works in both bull (buy fractal support in uptrend) and bear (sell fractal resistance in downtrend)
# - Volume filter (1.5x average) ensures institutional participation
# - Position size 0.25 targets ~20-60 trades/year, avoiding fee drag on 6h timeframe
# - Uses actual Williams Fractal logic (not pivots) for unique edge in BTC/ETH markets