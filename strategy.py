#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_1d_camarilla_breakout_v1
# Uses daily Camarilla levels for directional bias, 4h for trend filter (ADX), and 1h for entry timing.
# Buys when price breaks above daily H3 with 4h ADX > 25 and volume confirmation during active session (08-20 UTC).
# Shorts when price breaks below daily L3 under same conditions.
# Designed for low trade frequency (target: 15-37/year) to minimize fee drift.
# Works in bull markets (continuation breakouts) and bear markets (continuation breakdowns).

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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # already datetime64[ms], .hour works
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    
    # Align daily levels to 1h timeframe
    h3_level = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range for 4h
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus and Minus Directional Movement
    plus_dm = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    minus_dm = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_4h = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    plus_di = np.where(atr_4h != 0, 100 * plus_dm_smooth / atr_4h, 0)
    minus_di = np.where(atr_4h != 0, 100 * minus_dm_smooth / atr_4h, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_4h = wilders_smooth(dx, 14)
    
    # Align 4h ADX to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    adx_filter = adx_4h_aligned > 25
    
    # Volume confirmation: volume > 2.0 * 20-period average (1h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h3_level[i]) or np.isnan(l3_level[i]) or np.isnan(adx_filter[i]):
            signals[i] = 0.0
            continue
        
        # Require session, volume, and strong trend filters
        if not (session_mask[i] and vol_confirm[i] and adx_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above daily H3 with volume
        if close[i] > h3_level[i] and position != 1:
            position = 1
            signals[i] = 0.20
        # Short signal: price breaks below daily L3 with volume
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