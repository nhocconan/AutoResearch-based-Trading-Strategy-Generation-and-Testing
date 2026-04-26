#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: Camarilla pivot R1/S1 breakouts on 1h with 4h trend filter (EMA50) and volume spike (2x MA20) capture institutional moves. 
Long when price > R1 + volume spike + 4h uptrend; short when price < S1 + volume spike + 4h downtrend. 
Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years.
"""

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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (precomputed)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    uptrend_4h = close > ema_50_4h_aligned
    downtrend_4h = close < ema_50_4h_aligned
    
    # Calculate Camarilla pivots from previous day
    # Need daily high/low/close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align to 1h
    prev_high_1h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_1h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_1h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla levels
    range_ = prev_high_1h - prev_low_1h
    R1 = prev_close_1h + range_ * 1.1 / 12
    S1 = prev_close_1h - range_ * 1.1 / 12
    
    # Volume spike: volume > 2x MA20
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price > R1 + volume spike + 4h uptrend
            if close[i] > R1[i] and volume_spike[i] and uptrend_4h[i]:
                signals[i] = 0.20
                position = 1
            # Short: price < S1 + volume spike + 4h downtrend
            elif close[i] < S1[i] and volume_spike[i] and downtrend_4h[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price < S1 OR 4h trend changes to downtrend
            if close[i] < S1[i] or not uptrend_4h[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price > R1 OR 4h trend changes to uptrend
            if close[i] > R1[i] or not downtrend_4h[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0