#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Donchian breakout calculation, 1w for weekly pivot trend filter.
- Entry: Long when price breaks above 6h Donchian(20) high AND price > weekly pivot (from prior week) AND volume > 1.5x 20-period average.
         Short when price breaks below 6h Donchian(20) low AND price < weekly pivot AND volume > 1.5x 20-period average.
- Exit: Opposite Donchian breakout (price crosses midline) OR weekly pivot trend fails.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear breakout levels with built-in trend filtering.
- Weekly pivot adds higher-timeframe structural bias (bull/bear regime).
- Volume confirmation ensures breakouts have participation, reducing false signals.
- Works in bull markets (buy breakouts above weekly pivot) and bear markets (sell breakdowns below weekly pivot).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian(20) channels
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d weekly pivot (using prior week's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Weekly pivot: (Prior week HIGH + LOW + CLOSE) / 3
    weekly_high = df_1d['high'].values
    weekly_low = df_1d['low'].values
    weekly_close = df_1d['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot, additional_delay_bars=1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)  # Donchian(20) + volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Exit conditions: opposite Donchian breakout OR weekly pivot trend fails
        if position != 0:
            # Exit long: price falls below Donchian midline OR closes below weekly pivot
            if position == 1:
                if curr_close < donchian_mid[i] or curr_close < weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above Donchian midline OR closes above weekly pivot
            elif position == -1:
                if curr_close > donchian_mid[i] or curr_close > weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and weekly pivot alignment
        if position == 0:
            # Long: break above Donchian high AND above weekly pivot AND volume confirmed
            if curr_high > donchian_high[i] and curr_close > weekly_pivot_aligned[i] and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND below weekly pivot AND volume confirmed
            elif curr_low < donchian_low[i] and curr_close < weekly_pivot_aligned[i] and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0