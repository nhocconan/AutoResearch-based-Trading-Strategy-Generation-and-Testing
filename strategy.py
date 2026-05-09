# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Hypothesis: 6h timeframe with weekly Donchian breakout + monthly ATR filter.
# Uses weekly Donchian(20) channel from 1w data for structural breakouts.
# Monthly ATR(14) filters for volatility regime - only trade when volatility is elevated.
# Combines trend following with volatility filter to reduce whipsaw in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.
# Weekly data changes slowly, reducing whipsaw and improving win rate in both bull and bear markets.

name = "6h_Donchian20_1wATR14_Volatility_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) - highest high and lowest low over 20 weeks
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper band (20-period high)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian lower band (20-period low)
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Get monthly data for ATR filter (using 1w data as proxy for monthly - 4 weeks ~ 1 month)
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate True Range for weekly data
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.roll(high_1w, 1))
    low_close = np.abs(low_1w - np.roll(low_1w, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First value
    
    # ATR(14) on weekly data
    atr_14_1w = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # ATR filter: only trade when current ATR is above its 20-period average (elevated volatility)
    atr_ma = pd.Series(atr_14_1w_aligned).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr_14_1w_aligned > atr_ma
    
    # Breakout conditions
    breakout_up = close > donchian_high_aligned
    breakout_down = close < donchian_low_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + elevated volatility
            if breakout_up[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low + elevated volatility
            elif breakout_down[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian low or volatility drops
            if close[i] <= donchian_low_aligned[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian high or volatility drops
            if close[i] >= donchian_high_aligned[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals