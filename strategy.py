#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Volume_Filter_v1
Hypothesis: On 4h timeframe, buy when price breaks above Camarilla H3 resistance or sells below L3 support from the prior 1-day pivot calculation, with volume confirmation (>1.5x 20-period average). Uses 1-day ADX(14) > 20 to confirm trending regime and avoid false breakouts in chop. Designed for 20-40 trades/year by requiring confluence of price level break, volume surge, and trend strength. Works in bull markets via upward breaks and in bear markets via downward breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Breakout_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla levels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day OHLC for Camarilla
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's values (shift by 1 to avoid look-ahead)
    # Use roll to shift, first value will be NaN but handled by alignment
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels for each day
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    camarilla_h3 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev)
    camarilla_l3 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev)
    
    # Align Camarilla levels to 4h timeframe (will use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1-day ADX(14) for trend filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = np.nan  # First value has no previous close
        
        # Directional Movement
        up_move = np.diff(high, prepend=high[0])
        down_move = -np.diff(low, prepend=low[0])
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.nansum(arr[:period]) / period
            # Subsequent values smoothed
            for i in range(period, len(arr)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
                else:
                    result[i] = np.nan
            return result
        
        atr = smooth_wilder(tr, period)
        plus_di = 100 * smooth_wilder(plus_dm, period) / atr
        minus_di = 100 * smooth_wilder(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = smooth_wilder(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume average (20 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_h3_aligned[i]
        breakout_short = close[i] < camarilla_l3_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_1d_aligned[i] > 20
        
        # Entry conditions
        long_entry = breakout_long and volume_spike and trending
        short_entry = breakout_short and volume_spike and trending
        
        # Exit conditions: price returns to midpoint between H3 and L3
        midpoint = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals