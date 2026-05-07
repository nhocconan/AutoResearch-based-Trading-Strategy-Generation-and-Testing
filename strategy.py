#!/usr/bin/env python3
# 12H_Camarilla_R3_S3_WeeklyTrend_1DayVolume
# Hypothesis: 12-hour Camarilla R3/S3 level breakout with weekly trend filter (price > weekly EMA20) and daily volume spike confirmation.
# Uses weekly trend to avoid counter-trend trades in both bull and bear markets.
# Volume spike ensures momentum confirmation. Targets 15-35 trades/year to minimize fee drag.
# Uses discrete position sizing (0.25).

name = "12H_Camarilla_R3_S3_WeeklyTrend_1DayVolume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough data for EMA20
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for volume MA
        return np.zeros(n)
    
    # Calculate daily volume MA20
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate previous day's Camarilla levels (R3, S3)
    # Based on previous day's high, low, close
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    # Camarilla calculations
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    pivot = (prev_high + prev_low + prev_close) / 3
    rang = prev_high - prev_low
    r3 = prev_close + rang * 1.1 / 4
    s3 = prev_close - rang * 1.1 / 4
    
    # Volatility filter: avoid low volatility periods (ATR < 0.3% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.003 * close  # ATR > 0.3% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Ensure we have volume MA and previous day data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or vol_ma_1d_aligned[i] == 0 or
            np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(vol_filter[i]) or not vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.0x daily average volume)
        volume_filter = volume[i] > 2.0 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 + weekly uptrend + volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_20_1w_aligned[i] and   # Weekly uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + weekly downtrend + volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_20_1w_aligned[i] and   # Weekly downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to the pivot level (mean reversion)
            at_pivot = abs(close[i] - pivot[i]) < rang[i] * 0.1  # Within 10% of range
            
            if at_pivot:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals