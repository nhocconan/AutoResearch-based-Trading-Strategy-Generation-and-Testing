#!/usr/bin/env python3
"""
4h_12h_camarilla_breakout_volume
Strategy: 4h breakout with 12h Camarilla confluence
Timeframe: 4h
Leverage: 1.0
Hypothesis: Buy when 4h closes above 12h R3 with volume confirmation and 12h uptrend; sell when 4h closes below 12h S3 with volume confirmation and 12h downtrend. Uses 12h close for trend filter to avoid counter-trend trades. Designed to work in both bull and bear markets by only taking trades in direction of higher timeframe trend (12h close > prior 12h close for longs, < for shorts). Low-frequency design targets 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 4h ATR for volatility filter (optional, not used in entry but good for context)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 12h Close (trend filter: use prior 12h's close) ===
    close_12h = df_12h['close'].values
    # Trend: today's close > yesterday's close for uptrend, < for downtrend
    # We'll use the 12h close shifted by 1 to represent "prior 12h close" for trend
    close_12h_shifted = np.roll(close_12h, 1)
    close_12h_shifted[0] = np.nan
    close_12h_trend = align_htf_to_ltf(prices, df_12h, close_12h_shifted)
    
    # === 12h Camarilla (entry levels from prior 12h) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Prior 12h's Camarilla levels (use shifted values to avoid look-ahead)
    high_12h_shift = np.roll(high_12h, 1)
    low_12h_shift = np.roll(low_12h, 1)
    close_12h_shift = np.roll(close_12h, 1)
    high_12h_shift[0] = np.nan
    low_12h_shift[0] = np.nan
    close_12h_shift[0] = np.nan
    
    pivot_12h = (high_12h_shift + low_12h_shift + close_12h_shift) / 3
    range_12h = high_12h_shift - low_12h_shift
    r3_12h = close_12h_shift + range_12h * 1.166
    s3_12h = close_12h_shift - range_12h * 1.166
    
    # Align 12h Camarilla to 4h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Session filter: 0-23 UTC (covers major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(close_12h_trend[i]) or np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 4h volume must be elevated
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: price close vs prior 12h close (12h trend)
        uptrend_12h = price_close > close_12h_trend[i]
        downtrend_12h = price_close < close_12h_trend[i]
        
        # Long conditions: 4h closes above prior 12h's R3 with volume + 12h uptrend
        long_signal = volume_confirmed and (price_close > r3_12h_aligned[i]) and uptrend_12h
        
        # Short conditions: 4h closes below prior 12h's S3 with volume + 12h downtrend
        short_signal = volume_confirmed and (price_close < s3_12h_aligned[i]) and downtrend_12h
        
        # Exit when price returns to the 12h pivot (mean reversion within prior 12h's range)
        pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
        exit_long = position == 1 and price_close < pivot_12h_aligned[i]
        exit_short = position == -1 and price_close > pivot_12h_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
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

# Hypothesis: Buy when 4h closes above 12h R3 with volume confirmation and 12h uptrend; sell when 4h closes below 12h S3 with volume confirmation and 12h downtrend. Uses 12h close for trend filter to avoid counter-trend trades. Designed to work in both bull and bear markets by only taking trades in direction of higher timeframe trend (12h close > prior 12h close for longs, < for shorts). Low-frequency design targets 20-50 trades/year to minimize fee drag.