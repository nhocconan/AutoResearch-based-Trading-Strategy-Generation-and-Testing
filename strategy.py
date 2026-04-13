#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h/1d CAMARILLA PIVOT levels with volume confirmation and session filter.
CAMARILLA PIVOT calculates support/resistance levels based on previous day's range.
Long when price breaks above H4 resistance with volume spike during active session (08-20 UTC).
Short when price breaks below L4 support with volume spike during active session.
Uses 4h trend filter (price > 4h EMA20) to avoid counter-trend trades.
Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
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
    
    # Get 1d data for CAMARILLA pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate CAMARILLA pivot levels from previous day
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.25 * (High - Low)
    # L3 = Close - 1.25 * (High - Low)
    # H2 = Close + 1.083 * (High - Low)
    # L2 = Close - 1.083 * (High - Low)
    # H1 = Close + 1.0/1.1 * (High - Low)
    # L1 = Close - 1.0/1.1 * (High - Low)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Handle first value (no previous day)
    prev_close[0] = df_1d['close'].iloc[0]
    prev_high[0] = df_1d['high'].iloc[0]
    prev_low[0] = df_1d['low'].iloc[0]
    
    # Calculate pivot levels
    range_hl = prev_high - prev_low
    h4 = prev_close + 1.5 * range_hl
    l4 = prev_close - 1.5 * range_hl
    h3 = prev_close + 1.25 * range_hl
    l3 = prev_close - 1.25 * range_hl
    
    # Align H4 and L4 levels to 1h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Get 4h data for trend filter (EMA20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    ema_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.5x 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema_20 * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or 
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: price breaks above H4 resistance + volume spike + session + uptrend
        long_breakout = close[i] > h4_aligned[i]
        long_volume = vol_spike[i]
        long_session = session_filter[i]
        long_trend = close[i] > ema_4h_aligned[i]  # Above 4h EMA = uptrend
        
        # Short: price breaks below L4 support + volume spike + session + downtrend
        short_breakout = close[i] < l4_aligned[i]
        short_volume = vol_spike[i]
        short_session = session_filter[i]
        short_trend = close[i] < ema_4h_aligned[i]  # Below 4h EMA = downtrend
        
        long_entry = long_breakout and long_volume and long_session and long_trend
        short_entry = short_breakout and short_volume and short_session and short_trend
        
        # Exit when price returns to opposite level (mean reversion within CAMARILLA range)
        exit_long = position == 1 and close[i] < l4_aligned[i]
        exit_short = position == -1 and close[i] > h4_aligned[i]
        
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

name = "1h_1d_camarilla_pivot_volume_session"
timeframe = "1h"
leverage = 1.0