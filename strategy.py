#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakouts of daily Camarilla R1/S1 levels filtered by 1-week EMA trend (10) and volume spikes,
# executed on 4h timeframe to reduce trade frequency and avoid overtrading. Designed to capture strong trending moves
# in both bull and bear markets by following higher-timeframe trend. Target: 20-40 trades/year (~80-160 total over 4 years)
# to stay within optimal trade frequency for 4h.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # 1d Camarilla levels (R1, S1) - calculated from previous day's range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla R1 and S1 from previous day's close and range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    r1 = prev_close + (prev_range * 1.1 / 12)
    s1 = prev_close - (prev_range * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (wait for 1d candle to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1w EMA trend filter (10-period) - higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    ema_10_1w = pd.Series(df_1w['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour  # pre-compute before loop
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_10_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R1, 1w EMA uptrend, volume confirmation, session active
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_10_1w_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S1, 1w EMA downtrend, volume confirmation, session active
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_10_1w_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below 1d Camarilla S1 OR 1w EMA turns down
            if (close[i] < s1_aligned[i] or 
                close[i] < ema_10_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above 1d Camarilla R1 OR 1w EMA turns up
            if (close[i] > r1_aligned[i] or 
                close[i] > ema_10_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals