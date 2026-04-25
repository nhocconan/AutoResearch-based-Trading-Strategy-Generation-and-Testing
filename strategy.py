#!/usr/bin/env python3
"""
6h_ADX_DMI_Trend_Strength_With_Weekly_Pivot_Direction
Hypothesis: Trade strong ADX trends (ADX>25) in direction of weekly pivot bias (price vs weekly pivot).
ADX filters weak/choppy markets, weekly pivot provides structural bias.
Only trade long when price > weekly pivot and ADX rising, short when price < weekly pivot and ADX rising.
Uses discrete sizing 0.25 to limit drawdown in bear markets.
Target: 12-30 trades/year on 6h timeframe to minimize fee drag.
"""

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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point (standard formula: (H+L+C)/3)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate ADX on 6h timeframe (primary)
    # +DI, -DI, DX calculation
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    # Pad arrays to align with original indices
    plus_dm_padded = np.concatenate([[np.nan], plus_dm])
    minus_dm_padded = np.concatenate([[np.nan], minus_dm])
    tr_padded = np.concatenate([[np.nan], tr])
    
    atr = wilders_smoothing(tr_padded, period)
    plus_di = 100 * wilders_smoothing(plus_dm_padded, period) / atr
    minus_di = 100 * wilders_smoothing(minus_dm_padded, period) / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Ensure arrays align with prices index (adjust for padding)
    # plus_di, minus_di, adx already aligned due to padding logic
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after ADX warmup period
    start_idx = 2 * period  # Need enough for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(plus_di[i]) or np.isnan(minus_di[i])):
            signals[i] = 0.0
            continue
        
        # ADX trend strength filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: price above weekly pivot, strong trend, and +DI > -DI (bullish momentum)
            long_setup = (close[i] > weekly_pivot_aligned[i]) and \
                         strong_trend and \
                         (plus_di[i] > minus_di[i])
            # Short: price below weekly pivot, strong trend, and -DI > +DI (bearish momentum)
            short_setup = (close[i] < weekly_pivot_aligned[i]) and \
                          strong_trend and \
                          (minus_di[i] > plus_di[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: trend weakens (ADX < 20) OR momentum reverses (-DI > +DI) OR price crosses below pivot
            if (adx[i] < 20) or \
               (minus_di[i] > plus_di[i]) or \
               (close[i] < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: trend weakens (ADX < 20) OR momentum reverses (+DI > -DI) OR price crosses above pivot
            if (adx[i] < 20) or \
               (plus_di[i] > minus_di[i]) or \
               (close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_DMI_Trend_Strength_With_Weekly_Pivot_Direction"
timeframe = "6h"
leverage = 1.0