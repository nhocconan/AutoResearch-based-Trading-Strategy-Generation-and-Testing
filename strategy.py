#!/usr/bin/env python3
"""
1h_4h_1d_camarilla_breakout_volume
Strategy: 1h breakout with 4h/1d Camarilla confluence
Timeframe: 1h
Leverage: 1.0
Hypothesis: Buy when 1h closes above 4h R3 with volume confirmation and 1d uptrend; sell when 1h closes below 4h S3 with 1d downtrend. Uses 4h Camarilla for entry levels and 1d close for trend filter to avoid counter-trend trades. Designed to work in both bull and bear markets by only taking trades in direction of higher timeframe trend (1d close > prior 1d close for longs, < for shorts). Low-frequency design targets 15-37 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_volume"
timeframe = "1h"
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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 1h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d Close (trend filter: use prior 1d's close) ===
    close_1d = df_1d['close'].values
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan
    close_1d_trend = align_htf_to_ltf(prices, df_1d, close_1d_shifted)
    
    # === 4h Camarilla (entry levels from prior 4h) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Prior 4h's Camarilla levels (use shifted values to avoid look-ahead)
    high_4h_shift = np.roll(high_4h, 1)
    low_4h_shift = np.roll(low_4h, 1)
    close_4h_shift = np.roll(close_4h, 1)
    high_4h_shift[0] = np.nan
    low_4h_shift[0] = np.nan
    close_4h_shift[0] = np.nan
    
    pivot_4h = (high_4h_shift + low_4h_shift + close_4h_shift) / 3
    range_4h = high_4h_shift - low_4h_shift
    r3_4h = close_4h_shift + range_4h * 1.166
    s3_4h = close_4h_shift - range_4h * 1.166
    
    # Align 4h Camarilla to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or
            np.isnan(close_1d_trend[i]) or np.isnan(atr_1h[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 1h volume must be elevated
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: price close vs prior 1d close (1d trend)
        uptrend_1d = price_close > close_1d_trend[i]
        downtrend_1d = price_close < close_1d_trend[i]
        
        # Long conditions: 1h closes above prior 4h's R3 with volume + 1d uptrend
        long_signal = volume_confirmed and (price_close > r3_4h_aligned[i]) and uptrend_1d
        
        # Short conditions: 1h closes below prior 4h's S3 with volume + 1d downtrend
        short_signal = volume_confirmed and (price_close < s3_4h_aligned[i]) and downtrend_1d
        
        # Exit when price returns to the 4h pivot (mean reversion within prior 4h's range)
        pivot_4h = (high_4h_shift + low_4h_shift + close_4h_shift) / 3
        pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
        exit_long = position == 1 and price_close < pivot_4h_aligned[i]
        exit_short = position == -1 and price_close > pivot_4h_aligned[i]
        
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
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Buy when 1h closes above 4h R3 with volume confirmation and 1d uptrend; sell when 1h closes below 4h S3 with 1d downtrend. Uses 4h Camarilla for entry levels and 1d close for trend filter to avoid counter-trend trades. Designed to work in both bull and bear markets by only taking trades in direction of higher timeframe trend (1d close > prior 1d close for longs, < for shorts). Low-frequency design targets 15-37 trades/year to minimize fee drag.