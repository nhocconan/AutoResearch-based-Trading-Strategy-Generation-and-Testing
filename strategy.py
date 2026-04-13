#!/usr/bin/env python3
"""
4h_1d_1w_CamarillaBreakout_TrendFilter_v2
Hypothesis: Price breaking above daily R4 or below daily S4 Camarilla pivot levels with weekly volume expansion and filtered by weekly ADX trend strength captures strong momentum moves in both bull and bear markets. Daily pivots act as strong support/resistance; breaks indicate institutional interest. Weekly volume and trend filters reduce false breakouts. Target: 20-40 trades/year by requiring confluence.
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
    
    # Get daily data for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot points
    # Correct formula: R4 = Close + ((High-Low) * 1.1), S4 = Close - ((High-Low) * 1.1)
    diff = high_1d - low_1d
    camarilla_r4 = close_1d + (diff * 1.1)
    camarilla_s4 = close_1d - (diff * 1.1)
    
    # Get weekly data for volume and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly volume expansion: current volume > 1.5x 12-period average (~3 months)
    vol_ma_12 = pd.Series(volume_1w).rolling(window=12, min_periods=12).mean().values
    volume_expansion = volume_1w > (vol_ma_12 * 1.5)
    
    # Calculate ADX (14-period) for trend strength
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    
    up_move = np.where(high_1w - np.roll(high_1w, 1) > 0, high_1w - np.roll(high_1w, 1), 0)
    down_move = np.where(np.roll(low_1w, 1) - low_1w > 0, np.roll(low_1w, 1) - low_1w, 0)
    up_move[0] = 0
    down_move[0] = 0
    
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    period = 14
    atr = wilders_smooth(tr, period)
    plus_dm = wilders_smooth(up_move, period)
    minus_dm = wilders_smooth(down_move, period)
    
    plus_di = 100 * plus_dm / atr
    minus_di = 100 * minus_dm / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, period)
    strong_trend = adx > 20  # Moderate trend filter to allow more trades
    
    # Align all signals to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1w, volume_expansion.astype(float))
    strong_trend_aligned = align_htf_to_ltf(prices, df_1w, strong_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i]) or 
            np.isnan(strong_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Break of daily R4/S4 with weekly volume and trend
        long_break = close[i] > r4_aligned[i]
        short_break = close[i] < s4_aligned[i]
        
        long_entry = long_break and volume_expansion_aligned[i] > 0.5 and strong_trend_aligned[i] > 0.5
        short_entry = short_break and volume_expansion_aligned[i] > 0.5 and strong_trend_aligned[i] > 0.5
        
        # Exit when price returns to daily pivot point (mean reversion)
        pivot_1d = (high_1d + low_1d + close_1d) / 3.0
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
        exit_long = position == 1 and close[i] <= pivot_aligned[i]
        exit_short = position == -1 and close[i] >= pivot_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_CamarillaBreakout_TrendFilter_v2"
timeframe = "4h"
leverage = 1.0