#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1w_1d_camarilla_volume_trend_v1
# Uses weekly high/low to calculate daily Camarilla levels for the next week.
# Buys when price breaks above daily H3 with volume confirmation and 12h ADX > 20 (trending).
# Shorts when price breaks below daily L3 with volume confirmation and 12h ADX > 20.
# Uses 12h volume > 1.5 * 50-period average and ADX > 20 to filter for trending markets.
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag.
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation).

name = "12h_1w_1d_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    
    # Align to 12h timeframe (weekly levels update only after weekly bar closes)
    h3_level = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Get daily data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Volume confirmation: volume > 1.5 * 50-period average (daily)
    vol_ma_d = pd.Series(df_1d['volume'].values).rolling(window=50, min_periods=50).mean().values
    vol_ma_d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_d)
    vol_confirm = volume > (vol_ma_d_aligned * 1.5)
    
    # ADX trend filter: only trade when ADX > 20 (trending market) - daily
    # Calculate True Range for daily data
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr_d = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus and Minus Directional Movement for daily
    plus_dm = np.where((high_d[1:] - high_d[:-1]) > (low_d[:-1] - low_d[1:]), np.maximum(high_d[1:] - high_d[:-1], 0), 0)
    minus_dm = np.where((low_d[:-1] - low_d[1:]) > (high_d[1:] - high_d[:-1]), np.maximum(low_d[:-1] - low_d[1:], 0), 0)
    
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
    
    atr_d = wilders_smooth(tr_d, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(atr_d != 0, 100 * plus_dm_smooth / atr_d, 0)
    minus_di = np.where(atr_d != 0, 100 * minus_dm_smooth / atr_d, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_d = wilders_smooth(dx, 14)
    adx_d_aligned = align_htf_to_ltf(prices, df_1d, adx_d)
    adx_filter = adx_d_aligned > 20  # trending market only
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if levels or filters not ready
        if np.isnan(h3_level[i]) or np.isnan(l3_level[i]) or np.isnan(vol_confirm[i]) or np.isnan(adx_filter[i]):
            signals[i] = 0.0
            continue
        
        # Require both volume and trend filters
        if not (vol_confirm[i] and adx_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above daily H3 with volume and trend
        if close[i] > h3_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below daily L3 with volume and trend
        elif close[i] < l3_level[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif close[i] < l3_level[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > h3_level[i] and position == -1:
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