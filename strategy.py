#!/usr/bin/env python3
name = "6h_WilFractal_1dTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Williams fractals (need extra delay for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams fractal needs 2 extra 1d bars after center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 6h volume spike: > 2x 24-period average (48h lookback)
    vol_ma_6h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike_6h = volume > 2.0 * vol_ma_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for fractals and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish fractal confirmed + price above EMA34 + volume spike
            if (bullish_fractal_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                vol_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal confirmed + price below EMA34 + volume spike
            elif (bearish_fractal_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  vol_spike_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bearish fractal or price below EMA34
            if bearish_fractal_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bullish fractal or price above EMA34
            if bullish_fractal_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Williams fractal identifies swing highs/lows with confirmation delay.
# Long on bullish fractal + uptrend (price > EMA34) + volume spike.
# Short on bearish fractal + downtrend (price < EMA34) + volume spike.
# Exit on opposite fractal or trend violation.
# Position size 0.25 limits risk. Target ~25-40 trades/year.