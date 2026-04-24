#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d Bollinger Band width regime filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for volatility regime (Bollinger Band width).
- Bollinger Band width < 0.05 indicates low volatility (squeeze) → breakout strategy.
- Bollinger Band width > 0.10 indicates high volatility → mean reversion at Donchian mid.
- Entry: Long when price breaks above Donchian(20) upper AND BBwidth < 0.05 (bullish breakout in low vol).
         Short when price breaks below Donchian(20) lower AND BBwidth < 0.05 (bearish breakout in low vol).
         In high vol (BBwidth > 0.10): Long when price touches Donchian lower AND reverses up.
                                     Short when price touches Donchian upper AND reverses down.
- Exit: Opposite Donchian breakout or BBwidth regime shift.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Band width
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Bollinger Band width (20,2) on 1d
    close_1d = pd.Series(df_1d['close'])
    bb_middle = close_1d.rolling(window=20, min_periods=20).mean()
    bb_std = close_1d.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_values = bb_width.values
    
    # Align 1d Bollinger Band width to 4h
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width_values)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, lookback, 20)  # Need enough 1d bars for BBwidth and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_width_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bbwidth_val = bb_width_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if bbwidth_val < 0.05:  # Low volatility regime: breakout strategy
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > highest_high[i]:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below lower Donchian
                    elif curr_close < lowest_low[i]:
                        signals[i] = -0.25
                        position = -1
                elif bbwidth_val > 0.10:  # High volatility regime: mean reversion at extremes
                    # Long when price touches lower Donchian and shows reversal (close > low)
                    if curr_low <= lowest_low[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper Donchian and shows reversal (close < high)
                    elif curr_high >= highest_high[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
                # Middle volatility regime (0.05 <= BBwidth <= 0.10): no trading
        elif position == 1:
            # Long exit: price closes below Donchian mid OR BBwidth shifts to low volatility (breakout regime)
            if curr_close < donchian_mid[i] or bbwidth_val < 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR BBwidth shifts to low volatility (breakout regime)
            if curr_close > donchian_mid[i] or bbwidth_val < 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dBBWidthRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0