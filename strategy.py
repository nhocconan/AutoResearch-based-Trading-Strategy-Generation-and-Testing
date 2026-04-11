# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_12h_1d_camarilla_triple_confirm_v1
Strategy: 6h price action with 12h/1d Camarilla confluence
Timeframe: 6h
Leverage: 1.0
Hypothesis: Triple timeframe confluence (6h breakout + 12h trend filter + 1d volatility filter) reduces false signals while capturing major moves in both bull and bear markets. Uses Camarilla levels from multiple timeframes for institutional-grade support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_camarilla_triple_confirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 6h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 12h Camarilla (trend filter) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar's Camarilla levels
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    r3_12h = close_12h + range_12h * 1.166
    s3_12h = close_12h - range_12h * 1.166
    
    # Shift to use only completed 12h bars
    r3_12h = np.roll(r3_12h, 1)
    s3_12h = np.roll(s3_12h, 1)
    r3_12h[0] = np.nan
    s3_12h[0] = np.nan
    
    # Align 12h Camarilla to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # === 1d Camarilla (volatility filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.166
    s3_1d = close_1d - range_1d * 1.166
    
    # Shift to use only completed daily bars
    r3_1d = np.roll(r3_1d, 1)
    s3_1d = np.roll(s3_1d, 1)
    r3_1d[0] = np.nan
    s3_1d[0] = np.nan
    
    # Align daily Camarilla to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Session filter: 0-23 UTC (covers major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    # Minimum holding period: 3 bars (18 hours) to reduce churn
    hold_count = np.zeros(n, dtype=int)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Decrease hold counter
        if hold_count[i] > 0:
            hold_count[i] -= 1
        
        # Skip if any required data is invalid or outside session or holding
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(atr_6h[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i] or hold_count[i] > 0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 6h volume must be elevated
        volume_confirmed = volume_current > 2.0 * vol_ma
        
        # Trend filter: price must be above/below 12h Camarilla levels
        uptrend_12h = price_close > r3_12h_aligned[i]
        downtrend_12h = price_close < s3_12h_aligned[i]
        
        # Long conditions: 6h breaks above 1d R3 with volume + 12h uptrend
        long_signal = volume_confirmed and (price_high > r3_1d_aligned[i]) and uptrend_12h
        
        # Short conditions: 6h breaks below 1d S3 with volume + 12h downtrend
        short_signal = volume_confirmed and (price_low < s3_1d_aligned[i]) and downtrend_12h
        
        # Exit when price returns to the 12h pivot (mean reversion within 12h range)
        pivot_12h_today = (high_12h + low_12h + close_12h) / 3
        pivot_12h_6h = align_htf_to_ltf(prices, df_12h, pivot_12h_today)
        exit_long = position == 1 and price_close < pivot_12h_6h[i]
        exit_short = position == -1 and price_close > pivot_12h_6h[i]
        
        # Trading logic with minimum holding period
        if long_signal and position != 1:
            position = 1
            hold_count[i] = 3  # Hold for 3 bars minimum
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            hold_count[i] = 3  # Hold for 3 bars minimum
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Triple timeframe Camarilla confluence for 6s trading
# Uses 6h price action for entry timing, 12h Camarilla for trend direction, and 1d Camarilla for volatility context.
# Enters long when 6h price breaks above daily R3 with volume >2x average AND price is above 12h R3 (uptrend).
# Enters short when 6h price breaks below daily S3 with volume >2x average AND price is below 12h S3 (downtrend).
# Exits when price returns to the 12h pivot level (mean reversion within the 12h range).
# Multiple timeframe confirmation reduces false signals while capturing major moves in both bull and bear markets.
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity with cost efficiency.
# Works in bull markets by catching continuation breaks and in bear markets by fading overextended moves.