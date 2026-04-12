#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_1d_camarilla_breakout_v1
# Uses 4h and 1d timeframes for directional bias and 1h for entry timing.
# Long when price breaks above 1d H3 (4h trend up) with volume confirmation.
# Short when price breaks below 1d L3 (4h trend down) with volume confirmation.
# Uses 4h ADX > 25 to filter for strong trends.
# Designed for low trade frequency (target: 15-37/year) to minimize fee drag.
# Session filter: only trade between 08:00-20:00 UTC.
# Position size: 0.20.

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    dt = pd.to_datetime(open_time)
    hours = dt.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h ADX for trend strength
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    plus_dm_4h = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    minus_dm_4h = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_4h = wilders_smooth(tr_4h, 14)
    plus_dm_smooth_4h = wilders_smooth(plus_dm_4h, 14)
    minus_dm_smooth_4h = wilders_smooth(minus_dm_4h, 14)
    
    plus_di_4h = np.where(atr_4h != 0, 100 * plus_dm_smooth_4h / atr_4h, 0)
    minus_di_4h = np.where(atr_4h != 0, 100 * minus_dm_smooth_4h / atr_4h, 0)
    dx_4h = np.where((plus_di_4h + minus_di_4h) != 0, 100 * np.abs(plus_di_4h - minus_di_4h) / (plus_di_4h + minus_di_4h), 0)
    adx_4h = wilders_smooth(dx_4h, 14)
    adx_filter_4h = adx_4h > 25  # strong trend only
    
    # Align 4h ADX filter to 1h
    adx_filter_1h = align_htf_to_ltf(prices, df_4h, adx_filter_4h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    close_1d = df_1d['close'].shift(1).values
    
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + range_1d * 1.1 / 4
    camarilla_l3 = close_1d - range_1d * 1.1 / 4
    
    # Align 1d Camarilla levels to 1h
    h3_level = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if not in session
        if not session_mask[i]:
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if levels not ready
        if np.isnan(h3_level[i]) or np.isnan(l3_level[i]) or np.isnan(adx_filter_1h[i]):
            signals[i] = 0.0
            continue
        
        # Require both volume and strong trend filters
        if not (vol_confirm[i] and adx_filter_1h[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above 1d H3 with volume and 4h trend up
        if close[i] > h3_level[i] and position != 1:
            position = 1
            signals[i] = 0.20
        # Short signal: price breaks below 1d L3 with volume and 4h trend down
        elif close[i] < l3_level[i] and position != -1:
            position = -1
            signals[i] = -0.20
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
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals