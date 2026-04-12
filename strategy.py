#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_v1
# Uses weekly high/low to calculate weekly Camarilla levels for the next week.
# Buys when price breaks above weekly H3 with volume confirmation on daily close.
# Shorts when price breaks below weekly L3 with volume confirmation on daily close.
# Uses weekly ADX > 25 to filter for strong trends, avoiding false signals in weak trends or ranges.
# Designed for low trade frequency (target: 7-25 trades/year) to minimize fee drift.
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation).

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
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
    
    # Calculate Camarilla levels from previous week
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    
    # Align to 1d timeframe (weekly levels update only after weekly bar closes)
    h3_level = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Volume confirmation: volume > 2.0 * 20-period average (1d timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    # Weekly ADX trend filter: only trade when ADX > 25 (strong trend)
    # Calculate True Range using weekly data
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Plus and Minus Directional Movement
    plus_dm = np.where((df_1w['high'] - df_1w['high'].shift(1)) > (df_1w['low'].shift(1) - df_1w['low']), 
                       np.maximum(df_1w['high'] - df_1w['high'].shift(1), 0), 0)
    minus_dm = np.where((df_1w['low'].shift(1) - df_1w['low']) > (df_1w['high'] - df_1w['high'].shift(1)), 
                        np.maximum(df_1w['low'].shift(1) - df_1w['low'], 0), 0)
    
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
    
    # Pad arrays to match price length for alignment
    tr_padded = np.full(n, np.nan)
    plus_dm_padded = np.full(n, np.nan)
    minus_dm_padded = np.full(n, np.nan)
    
    # Align weekly data to daily
    tr_aligned = align_htf_to_ltf(prices, df_1w, tr)
    plus_dm_aligned = align_htf_to_ltf(prices, df_1w, plus_dm)
    minus_dm_aligned = align_htf_to_ltf(prices, df_1w, minus_dm)
    
    atr = wilders_smooth(tr_aligned, 14)
    plus_dm_smooth = wilders_smooth(plus_dm_aligned, 14)
    minus_dm_smooth = wilders_smooth(minus_dm_aligned, 14)
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    adx_filter = adx > 25  # strong trend only
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h3_level[i]) or np.isnan(l3_level[i]) or np.isnan(adx_filter[i]):
            signals[i] = 0.0
            continue
        
        # Require both volume and strong trend filters
        if not (vol_confirm[i] and adx_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly H3 with volume
        if close[i] > h3_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly L3 with volume
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