#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_v2
Hypothesis: On 6h timeframe, price breaking Donchian(20) channels with weekly pivot point bias and volume confirmation provides robust breakout signals. Weekly pivot direction (based on prior week's close relative to pivot) filters breakouts to trade with the weekly bias, reducing false breakouts in ranging markets. Volume confirmation ensures breakout strength. Designed for low trade frequency (~15-25/year) to minimize fee drag while capturing strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly bias: 1 if close > pivot (bullish bias), -1 if close < pivot (bearish bias)
    weekly_bias = np.where(weekly_close > weekly_pivot, 1, -1)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Calculate Donchian channels (20-period) on 6h
    donchian_up = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_down = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), ATR(14), volume MA(20)
    start_idx = max(20, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_up[i]) or
            np.isnan(donchian_down[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(weekly_bias_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_confirmed = vol_ratio[i] > 1.5  # volume at least 1.5x average
        bias = weekly_bias_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND weekly bullish bias AND volume confirmation
            long_signal = (close_val > donchian_up[i]) and (bias == 1) and vol_confirmed
            
            # Short: price breaks below Donchian lower AND weekly bearish bias AND volume confirmation
            short_signal = (close_val < donchian_down[i]) and (bias == -1) and vol_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below Donchian lower OR ATR stoploss hit
            if (close_val < donchian_down[i]) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above Donchian upper OR ATR stoploss hit
            if (close_val > donchian_up[i]) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_v2"
timeframe = "6h"
leverage = 1.0