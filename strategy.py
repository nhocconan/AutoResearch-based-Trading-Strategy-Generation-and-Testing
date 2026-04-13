#!/usr/bin/env python3
"""
12h_1d_Williams_Fractal_Breakout
Hypothesis: Williams fractals on 1d provide high-probability reversal zones. 
Breakouts above bearish fractals or below bullish fractals with volume confirmation
indicate institutional momentum. Trades in direction of 1w EMA200 trend to avoid 
counter-trend whipsaws. Works in bull markets (breakouts continue) and bear markets 
(mean reversion at fractal levels). Targets 12-37 trades/year on 12h timeframe.
"""

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
    
    # Get daily data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Need 2 extra daily bars for fractal confirmation (per rule 2b)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: current volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean()
    volume_expansion = volume > (vol_ma_50 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Breakout above bearish fractal (resistance break)
        # 2. Volume expansion
        # 3. Above weekly EMA200 (bullish trend filter)
        breakout_long = (close[i] > bearish_fractal_aligned[i]) and volume_expansion[i]
        long_condition = breakout_long and (close[i] > ema_200_aligned[i])
        
        # Short conditions:
        # 1. Breakdown below bullish fractal (support break)
        # 2. Volume expansion
        # 3. Below weekly EMA200 (bearish trend filter)
        breakdown_short = (close[i] < bullish_fractal_aligned[i]) and volume_expansion[i]
        short_condition = breakdown_short and (close[i] < ema_200_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_Williams_Fractal_Breakout"
timeframe = "12h"
leverage = 1.0