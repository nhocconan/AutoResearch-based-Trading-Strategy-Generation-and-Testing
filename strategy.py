#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: 12h price breaks Camarilla R3/S3 levels from daily pivot, filtered by 1w EMA trend (21) and volume spikes.
# Designed to capture strong breakouts with trend alignment while avoiding false signals in chop.
# Works in bull/bear by following weekly trend. Target: 12-30 trades/year (~50-120 total over 4 years).

name = "12h_Camarilla_Pivot_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily high/low/close for Camarilla pivot (use previous day's values)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    R3 = pivot + (range_ * 1.1 / 2.0)  # R3 = pivot + 1.1*(H-L)/2
    S3 = pivot - (range_ * 1.1 / 2.0)  # S3 = pivot - 1.1*(H-L)/2
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Weekly EMA trend filter (21-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour  # pre-compute before loop
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, weekly EMA uptrend, volume confirmation, session active
            if (close[i] > R3_aligned[i] and 
                close[i] > ema_21_1w_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, weekly EMA downtrend, volume confirmation, session active
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema_21_1w_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S3 OR weekly EMA turns down
            if (close[i] < S3_aligned[i] or 
                close[i] < ema_21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R3 OR weekly EMA turns up
            if (close[i] > R3_aligned[i] or 
                close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals