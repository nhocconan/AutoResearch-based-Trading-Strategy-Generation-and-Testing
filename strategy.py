#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return signals
    
    # 4h high/low for Camarilla pivot calculation
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1h EMA for entry timing
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF data to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-calculate Camarilla levels for each 4h bar
    camarilla_levels = []
    for i in range(len(close_4h)):
        if i < 1:
            camarilla_levels.append({
                'h3': np.nan, 'l3': np.nan,
                'h4': np.nan, 'l4': np.nan
            })
            continue
        # Previous 4h bar's high/low/close
        ph = high_4h[i-1]
        pl = low_4h[i-1]
        pc = close_4h[i-1]
        range_val = ph - pl
        if range_val <= 0:
            camarilla_levels.append({
                'h3': np.nan, 'l3': np.nan,
                'h4': np.nan, 'l4': np.nan
            })
            continue
        camarilla_levels.append({
            'h3': pc + range_val * 1.1 / 6,
            'l3': pc - range_val * 1.1 / 6,
            'h4': pc + range_val * 1.1 / 4,
            'l4': pc - range_val * 1.1 / 4
        })
    
    # Align Camarilla levels to 1h timeframe
    h3_series = np.array([l['h3'] for l in camarilla_levels])
    l3_series = np.array([l['l3'] for l in camarilla_levels])
    h4_series = np.array([l['h4'] for l in camarilla_levels])
    l4_series = np.array([l['l4'] for l in camarilla_levels])
    
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3_series)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3_series)
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4_series)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4_series)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(100, n):
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20[i])):
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        price_close = close[i]
        ema_trend = ema_50_1d_aligned[i]
        ema_fast = ema_20[i]
        
        # Camarilla levels
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        
        # Trend filter: price relative to daily EMA50
        above_trend = price_close > ema_trend
        below_trend = price_close < ema_trend
        
        # Entry signals: Camarilla H3/L3 breakout with EMA20 confirmation
        long_signal = False
        short_signal = False
        
        # Long: price breaks above H3 and above EMA20
        if price_close > h3 and price_close > ema_fast and above_trend:
            long_signal = True
        
        # Short: price breaks below L3 and below EMA20
        if price_close < l3 and price_close < ema_fast and below_trend:
            short_signal = True
        
        # Exit: price returns to EMA20 (mean reversion)
        exit_long = price_close < ema_fast
        exit_short = price_close > ema_fast
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1h Camarilla breakout strategy with daily EMA50 trend filter.
# Enters long when price breaks above Camarilla H3 level (previous 4h bar) with EMA20 confirmation and above daily EMA50 trend.
# Enters short when price breaks below Camarilla L3 level with EMA20 confirmation and below daily EMA50 trend.
# Exits when price returns to EMA20 (mean reversion).
# Uses Camarilla levels from 4h timeframe for support/resistance structure.
# EMA20 on 1h for entry timing and exit signal.
# Daily EMA50 filter ensures trades align with higher timeframe trend.
# Session filter (08-20 UTC) reduces noise trades during low-volume periods.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Works in both bull and bear markets by trading breakouts in the direction of higher timeframe trend.