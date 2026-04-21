#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_WeeklyPivot_Filter_v1
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with 1d EMA50 trend and filtered by weekly pivot position (above/below weekly pivot) capture institutional moves with reduced false breakouts. Weekly pivot acts as a dynamic support/resistance filter that works in both bull and bear markets by identifying key levels where price reacts. Designed for low trade frequency (~15-30/year) to minimize fee drag on 6h chart.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 5:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Weekly pivot points (using prior week OHLC) ===
    # Weekly Pivot = (High + Low + Close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        pivot = weekly_pivot_aligned[i]
        trend_1d = ema_50_1d_aligned[i]
        
        # Calculate Donchian(20) on 6h data using rolling window
        if i >= 20:
            lookback_start = max(100, i - 19)  # Ensure we have enough warmup
            if lookback_start <= i:
                high_window = prices['high'].iloc[lookback_start:i+1]
                low_window = prices['low'].iloc[lookback_start:i+1]
                donchian_high = high_window.max()
                donchian_low = low_window.min()
            else:
                # Not enough lookback data yet
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian HIGH + price above 1d EMA50 + price above weekly pivot
            if price_close > donchian_high and price_close > trend_1d and price_close > pivot:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below Donchian LOW + price below 1d EMA50 + price below weekly pivot
            elif price_close < donchian_low and price_close < trend_1d and price_close < pivot:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit: price crosses 1d EMA50 in opposite direction
            if position == 1 and price_close < trend_1d:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_WeeklyPivot_Filter_v1"
timeframe = "6h"
leverage = 1.0