#!/usr/bin/env python3
"""
6h_1w_camarilla_trend_follow
Strategy: 6h price action with weekly Camarilla pivot trend following
Timeframe: 6h
Leverage: 1.0
Hypothesis: Trade in direction of weekly trend using price position relative to weekly Camarilla levels. 
Go long when 6h close is above weekly R3 and weekly trend is up (weekly close > prior weekly close).
Go short when 6h close is below weekly S3 and weekly trend is down (weekly close < prior weekly close).
Exit when price returns to weekly pivot. Uses volume confirmation to avoid false breakouts.
Designed to work in both bull and bear markets by following higher timeframe trend.
Targets 12-37 trades per year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_camarilla_trend_follow"
timeframe = "6h"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 6h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Weekly Close (trend filter: use prior week's close) ===
    close_1w = df_1w['close'].values
    # Trend: today's close > yesterday's close for uptrend, < for downtrend
    # We'll use the 1w close shifted by 1 to represent "prior week close" for trend
    close_1w_shifted = np.roll(close_1w, 1)
    close_1w_shifted[0] = np.nan
    close_1w_trend = align_htf_to_ltf(prices, df_1w, close_1w_shifted)
    
    # === Weekly Camarilla (entry levels from prior week) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Prior week's Camarilla levels (use shifted values to avoid look-ahead)
    high_1w_shift = np.roll(high_1w, 1)
    low_1w_shift = np.roll(low_1w, 1)
    close_1w_shift = np.roll(close_1w, 1)
    high_1w_shift[0] = np.nan
    low_1w_shift[0] = np.nan
    close_1w_shift[0] = np.nan
    
    pivot_1w = (high_1w_shift + low_1w_shift + close_1w_shift) / 3
    range_1w = high_1w_shift - low_1w_shift
    r3_1w = close_1w_shift + range_1w * 1.166
    s3_1w = close_1w_shift - range_1w * 1.166
    
    # Align weekly Camarilla to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Session filter: 0-23 UTC (covers major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(close_1w_trend[i]) or
            np.isnan(atr_6h[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 6h volume must be elevated
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: price close vs prior week close (1w trend)
        uptrend_1w = price_close > close_1w_trend[i]
        downtrend_1w = price_close < close_1w_trend[i]
        
        # Long conditions: 6h closes above prior week's R3 with volume + 1w uptrend
        long_signal = volume_confirmed and (price_close > r3_1w_aligned[i]) and uptrend_1w
        
        # Short conditions: 6h closes below prior week's S3 with volume + 1w downtrend
        short_signal = volume_confirmed and (price_close < s3_1w_aligned[i]) and downtrend_1w
        
        # Exit when price returns to the weekly pivot (mean reversion within prior week's range)
        exit_long = position == 1 and price_close < pivot_1w_aligned[i]
        exit_short = position == -1 and price_close > pivot_1w_aligned[i]
        
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

# Hypothesis: Trade in direction of weekly trend using price position relative to weekly Camarilla levels. 
# Go long when 6h close is above weekly R3 and weekly trend is up (weekly close > prior weekly close).
# Go short when 6h close is below weekly S3 and weekly trend is down (weekly close < prior weekly close).
# Exit when price returns to weekly pivot. Uses volume confirmation to avoid false breakouts.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Targets 12-37 trades per year to minimize fee drag.