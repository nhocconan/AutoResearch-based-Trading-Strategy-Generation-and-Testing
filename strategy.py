#!/usr/bin/env python3
"""
12h_1d_CamarillaPivot_Breakout_VolumeTrend
Hypothesis: Price breaking above/below daily Camarilla pivot levels (H4/L4) with volume expansion 
and filtered by daily ADX trend strength captures momentum moves in both bull and bear markets. 
Daily Camarilla levels act as strong intraday support/resistance; breaks indicate institutional 
interest. Using 12h timeframe reduces trade frequency to avoid fee drag while maintaining 
effectiveness. Target: 20-40 trades per year.
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
    
    # Get daily data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    # Formula: (H+L+C)/3 as base, then H4/L4 = C +/- 1.1*(H-L)
    camarilla_base = (high_1d + low_1d + close_1d) / 3.0
    h4 = camarilla_base + 1.1 * (high_1d - low_1d)
    l4 = camarilla_base - 1.1 * (high_1d - low_1d)
    
    # Volume expansion: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume_1d > (vol_ma_20 * 2.0)
    
    # Calculate ADX (14-period) for trend strength
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.where(np.arange(len(tr)) == 0, np.nan, tr)  # Set first element to NaN
    
    up_move = np.where(high_1d - np.roll(high_1d, 1) > 0, high_1d - np.roll(high_1d, 1), 0)
    down_move = np.where(np.roll(low_1d, 1) - low_1d > 0, np.roll(low_1d, 1) - low_1d, 0)
    up_move = np.where(np.arange(len(up_move)) == 0, 0, up_move)
    down_move = np.where(np.arange(len(down_move)) == 0, 0, down_move)
    
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # Initialize first value
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
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, period)
    strong_trend = adx > 25  # Strong trend filter
    
    # Align all signals to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1d, volume_expansion.astype(float))
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i]) or 
            np.isnan(strong_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Break of daily H4/L4 with volume and trend
        long_break = close[i] > h4_aligned[i]
        short_break = close[i] < l4_aligned[i]
        
        long_entry = long_break and volume_expansion_aligned[i] > 0.5 and strong_trend_aligned[i] > 0.5
        short_entry = short_break and volume_expansion_aligned[i] > 0.5 and strong_trend_aligned[i] > 0.5
        
        # Exit when price returns to camarilla base (mean reversion)
        camarilla_base_aligned = align_htf_to_ltf(prices, df_1d, camarilla_base)
        exit_long = position == 1 and close[i] <= camarilla_base_aligned[i]
        exit_short = position == -1 and close[i] >= camarilla_base_aligned[i]
        
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

name = "12h_1d_CamarillaPivot_Breakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0