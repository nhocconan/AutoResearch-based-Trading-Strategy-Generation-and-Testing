#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1w_1d_camarilla_breakout_v1
# Uses weekly and daily data to calculate weekly and daily Camarilla levels.
# Buys when price breaks above weekly H3 with volume confirmation and price > daily open.
# Shorts when price breaks below weekly L3 with volume confirmation and price < daily open.
# Uses ADX > 20 on 12h to filter for moderate trends, avoiding false signals in weak trends.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation).

name = "12h_1w_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for additional filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous week
    weekly_high_prev = df_1w['high'].shift(1).values
    weekly_low_prev = df_1w['low'].shift(1).values
    weekly_close_prev = df_1w['close'].shift(1).values
    
    # Calculate daily open for filter
    daily_open = df_1d['open'].values
    
    # Weekly Camarilla formulas
    weekly_range_prev = weekly_high_prev - weekly_low_prev
    weekly_h3 = weekly_close_prev + weekly_range_prev * 1.1 / 4
    weekly_l3 = weekly_close_prev - weekly_range_prev * 1.1 / 4
    
    # Align to 12h timeframe (weekly levels update only after weekly bar closes)
    weekly_h3_level = align_htf_to_ltf(prices, df_1w, weekly_h3)
    weekly_l3_level = align_htf_to_ltf(prices, df_1w, weekly_l3)
    
    # Align daily open to 12h timeframe
    daily_open_level = align_htf_to_ltf(prices, df_1d, daily_open)
    
    # Volume confirmation: volume > 1.5 * 30-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # ADX trend filter: only trade when ADX > 20 (moderate trend) on 12h
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus and Minus Directional Movement
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    # Wilder's smoothing function
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    adx_filter = adx > 20  # moderate trend only
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(weekly_h3_level[i]) or np.isnan(weekly_l3_level[i]) or np.isnan(daily_open_level[i]) or np.isnan(adx_filter[i]):
            signals[i] = 0.0
            continue
        
        # Require volume and moderate trend filters
        if not (vol_confirm[i] and adx_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly H3 with volume and price > daily open
        if close[i] > weekly_h3_level[i] and close[i] > daily_open_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly L3 with volume and price < daily open
        elif close[i] < weekly_l3_level[i] and close[i] < daily_open_level[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif close[i] < weekly_l3_level[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > weekly_h3_level[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals