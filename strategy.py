#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_camarilla_breakout_v1
# Uses daily high/low to calculate daily Camarilla levels for the next 12h bar.
# Buys when price breaks above daily H3 with volume confirmation.
# Shorts when price breaks below daily L3 with volume confirmation.
# Uses ADX > 25 on 1d timeframe to filter for strong trends, avoiding false signals in weak trends or ranges.
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag.
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation).

name = "12h_1d_camarilla_breakout_v1"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    
    # Align to 12h timeframe (daily levels update only after daily bar closes)
    h3_level = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 2.0 * 20-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    # ADX trend filter on 1d timeframe: only trade when ADX > 25 (strong trend)
    # Calculate True Range for daily data
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_daily = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus and Minus Directional Movement for daily data
    plus_dm_daily = np.where((df_1d['high'].values[1:] - df_1d['high'].values[:-1]) > (df_1d['low'].values[:-1] - df_1d['low'].values[1:]), np.maximum(df_1d['high'].values[1:] - df_1d['high'].values[:-1], 0), 0)
    minus_dm_daily = np.where((df_1d['low'].values[:-1] - df_1d['low'].values[1:]) > (df_1d['high'].values[1:] - df_1d['high'].values[:-1]), np.maximum(df_1d['low'].values[:-1] - df_1d['low'].values[1:], 0), 0)
    
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
    
    atr_daily = wilders_smooth(tr_daily, 14)
    plus_dm_smooth_daily = wilders_smooth(plus_dm_daily, 14)
    minus_dm_smooth_daily = wilders_smooth(minus_dm_daily, 14)
    
    # Avoid division by zero
    plus_di_daily = np.where(atr_daily != 0, 100 * plus_dm_smooth_daily / atr_daily, 0)
    minus_di_daily = np.where(atr_daily != 0, 100 * minus_dm_smooth_daily / atr_daily, 0)
    dx_daily = np.where((plus_di_daily + minus_di_daily) != 0, 100 * np.abs(plus_di_daily - minus_di_daily) / (plus_di_daily + minus_di_daily), 0)
    adx_daily = wilders_smooth(dx_daily, 14)
    adx_filter = adx_daily > 25  # strong trend only on daily
    
    # Align ADX filter to 12h timeframe
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h3_level[i]) or np.isnan(l3_level[i]) or np.isnan(adx_filter_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require both volume and strong trend filters
        if not (vol_confirm[i] and adx_filter_aligned[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above daily H3 with volume
        if close[i] > h3_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below daily L3 with volume
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