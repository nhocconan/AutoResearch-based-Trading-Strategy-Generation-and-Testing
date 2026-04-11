#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume_trend_v33
Strategy: 4h breakout with 1d Camarilla levels, volume confirmation, and trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses 1d Camarilla levels (S3/R3) as breakout levels with volume confirmation (2.0x 20-period average) and EMA trend filter (price > EMA20 for longs, < EMA20 for shorts). Designed to capture strong momentum moves while minimizing false breakouts. Target: 20-40 trades/year to avoid fee drag while maintaining edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_trend_v33"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA for trend filter
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).values
    
    # 4h volume filter: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d Close (prior close for context) ===
    close_1d = df_1d['close'].values
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan
    close_1d_prior = align_htf_to_ltf(prices, df_1d, close_1d_shifted)
    
    # === 1d Camarilla (entry levels from prior 1d) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    pivot_1d = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_1d = high_1d_shift - low_1d_shift
    r3_1d = close_1d_shift + range_1d * 1.166
    s3_1d = close_1d_shift - range_1d * 1.166
    
    # Align 1d Camarilla to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(close_1d_prior[i]) or np.isnan(ema_20[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_open = open_price[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 4h volume must be expanded
        volume_expanded = volume_current > 2.0 * vol_ma
        
        # Trend filter: price above/below EMA20
        uptrend = price_close > ema_20[i]
        downtrend = price_close < ema_20[i]
        
        # Long conditions: 4h closes above prior 1d's R3 with volume expansion + uptrend
        long_signal = volume_expanded and uptrend and (price_close > r3_1d_aligned[i])
        
        # Short conditions: 4h closes below prior 1d's S3 with volume expansion + downtrend
        short_signal = volume_expanded and downtrend and (price_close < s3_1d_aligned[i])
        
        # Exit when price returns to the 1d pivot (mean reversion within prior 1d's range)
        exit_long = position == 1 and price_close < pivot_1d_aligned[i]
        exit_short = position == -1 and price_close > pivot_1d_aligned[i]
        
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

# Hypothesis: Uses 1d Camarilla levels (S3/R3) as breakout levels with volume confirmation (2.0x 20-period average) and EMA trend filter (price > EMA20 for longs, < EMA20 for shorts). Designed to capture strong momentum moves while minimizing false breakouts. Target: 20-40 trades/year to avoid fee drag while maintaining edge in both bull and bear markets.