#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume_trend_v4
Strategy: 4h price action with 1d Camarilla confluence
Timeframe: 4h
Leverage: 1.0
Hypothesis: Buy when 4h breaks above daily R3 with volume confirmation and 1d trend filter; sell when breaks below daily S3 with volume confirmation and 1d trend filter. Uses 1d EMA50 for trend filter to avoid counter-trend trades. Designed to work in both bull and bear markets by only taking trades in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_trend_v4"
timeframe = "4h"
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
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA50 (trend filter) ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Camarilla (entry levels) ===
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
    
    # Align daily Camarilla to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Session filter: 0-23 UTC (covers major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    # Minimum holding period: 3 bars (12 hours) to reduce churn
    hold_count = np.zeros(n, dtype=int)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Decrease hold counter
        if hold_count[i] > 0:
            hold_count[i] -= 1
        
        # Skip if any required data is invalid or outside session or holding
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i] or hold_count[i] > 0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Volume confirmation: 4h volume must be elevated
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: price must be above/below 1d EMA50
        uptrend_1d = price_close > ema_50
        downtrend_1d = price_close < ema_50
        
        # Long conditions: 4h breaks above 1d R3 with volume + 1d uptrend
        long_signal = volume_confirmed and (price_high > r3_1d_aligned[i]) and uptrend_1d
        
        # Short conditions: 4h breaks below 1d S3 with volume + 1d downtrend
        short_signal = volume_confirmed and (price_low < s3_1d_aligned[i]) and downtrend_1d
        
        # Exit when price returns to the 1d pivot (mean reversion within 1d range)
        pivot_1d_today = (high_1d + low_1d + close_1d) / 3
        pivot_1d_4h = align_htf_to_ltf(prices, df_1d, pivot_1d_today)
        exit_long = position == 1 and price_close < pivot_1d_4h[i]
        exit_short = position == -1 and price_close > pivot_1d_4h[i]
        
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

# Hypothesis: Buy when 4h breaks above daily R3 with volume confirmation and 1d trend filter; sell when breaks below daily S3 with volume confirmation and 1d trend filter. Uses 1d EMA50 for trend filter to avoid counter-trend trades. Designed to work in both bull and bear markets by only taking trades in direction of higher timeframe trend.