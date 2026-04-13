#!/usr/bin/env python3
"""
4h_1d_Range_Breakout_With_Volume_Confirmation
Hypothesis: Trade breakouts from daily high/low ranges on 4h timeframe with volume confirmation and trend filter.
Daily ranges act as significant support/resistance levels. Breakouts above daily high indicate bullish momentum,
breakdowns below daily low indicate bearish momentum. Volume > 1.5x 20-period average confirms institutional participation.
Trend filter: price above/below 50-period EMA on 4h to avoid counter-trend trades.
Target: 25-35 trades/year to minimize fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for range calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align daily high/low to 4h
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Get 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above daily high with volume expansion and price above EMA50
        long_condition = (close[i] > daily_high_aligned[i]) and volume_expansion[i] and (close[i] > ema_50[i])
        
        # Short: breakdown below daily low with volume expansion and price below EMA50
        short_condition = (close[i] < daily_low_aligned[i]) and volume_expansion[i] and (close[i] < ema_50[i])
        
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

name = "4h_1d_Range_Breakout_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0