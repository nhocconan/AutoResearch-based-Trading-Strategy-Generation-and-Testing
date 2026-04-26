#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_TrendFilter_v2
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
- Long when price breaks above Donchian(20) high AND weekly pivot shows bullish bias (price > weekly pivot) AND volume > 1.5 * volume_ma(20)
- Short when price breaks below Donchian(20) low AND weekly pivot shows bearish bias (price < weekly pivot) AND volume > 1.5 * volume_ma(20)
- Weekly pivot provides structural bias from higher timeframe (1w) to avoid counter-trend trades
- Donchian(20) breakout captures momentum with defined risk/entry levels
- Volume confirmation filters breakouts with institutional participation
- Designed for low frequency (target 12-30 trades/year on 6h) to minimize fee drag in bear markets
- Novelty: Weekly pivot as trend filter (not commonly used) combined with Donchian breakouts for 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Using rolling window to get previous week's values
    wk_high = pd.Series(df_1w['high'].values).rolling(window=2, min_periods=1).max().shift(1).values
    wk_low = pd.Series(df_1w['low'].values).rolling(window=2, min_periods=1).min().shift(1).values
    wk_close = pd.Series(df_1w['close'].values).rolling(window=2, min_periods=1).mean().shift(1).values
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (wk_high + wk_low + wk_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Weekly bias: 1 = bullish (price > weekly pivot), -1 = bearish (price < weekly pivot), 0 = neutral
    weekly_bias = np.where(weekly_pivot_aligned > 0, 
                           np.where(close > weekly_pivot_aligned, 1, -1), 
                           0)
    
    # Calculate Donchian(20) channels on 6h chart (primary timeframe)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume filter: volume > 1.5 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_bias[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with weekly bias and volume spike filter
        if position == 0:
            # Long: Price breaks above Donchian high AND weekly bullish bias AND volume spike
            if close[i] > donchian_high[i] and weekly_bias[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND weekly bearish bias AND volume spike
            elif close[i] < donchian_low[i] and weekly_bias[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR weekly bias turns bearish
            if close[i] < donchian_low[i] or weekly_bias[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR weekly bias turns bullish
            if close[i] > donchian_high[i] or weekly_bias[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_TrendFilter_v2"
timeframe = "6h"
leverage = 1.0