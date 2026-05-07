#!/usr/bin/env python3
# 6h_WilliamsFractal_1dTrend_Volume
# Hypothesis: Uses daily Williams Fractals (with 2-bar confirmation) to identify potential reversal points, 
# filtered by 1d EMA34 trend direction and volume confirmation. Fractals provide high-probability 
# reversal zones, while EMA34 filters for trend alignment and volume reduces false signals. 
# Works in both bull and bear markets by trading reversals in the direction of the daily trend.
# Target: 12-30 trades/year to minimize fee drag while capturing meaningful moves.

name = "6h_WilliamsFractal_1dTrend_Volume"
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
    
    # Get 1d data for trend filter and fractal calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Fractals on 1d data (requires 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Apply 2-bar additional delay for confirmation (as per rule 2b)
    bearish_fractal_6h = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_6h = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate volume spike on 6h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_34_6h[i]) or np.isnan(bearish_fractal_6h[i]) or 
            np.isnan(bullish_fractal_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish fractal (potential support) + above 1d EMA34 + volume spike
            if bullish_fractal_6h[i] and close[i] > ema_34_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal (potential resistance) + below 1d EMA34 + volume spike
            elif bearish_fractal_6h[i] and close[i] < ema_34_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below 1d EMA34 (trend reversal) or bearish fractal appears
            if close[i] < ema_34_6h[i] or bearish_fractal_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above 1d EMA34 (trend reversal) or bullish fractal appears
            if close[i] > ema_34_6h[i] or bullish_fractal_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals