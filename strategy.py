#!/usr/bin/env python3
# 1h_4hTrend_1dLiquidity_Capture
# Hypothesis: 4h trend filter (EMA20) for direction, 1d liquidity sweep detection for entry timing.
# Enter long when 1h price breaks above recent 1d high with volume, in 4h uptrend.
# Enter short when price breaks below recent 1d low with volume, in 4h downtrend.
# Uses liquidity sweeps to capture institutional order flow, works in bull via trend-following
# and in bear via mean-reversion at liquidity zones. Target: 15-30 trades/year.

name = "1h_4hTrend_1dLiquidity_Capture"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h EMA20 for trend
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for liquidity levels (previous day high/low)
    df_1d = get_htf_data(prices, '1d')
    # Previous day high and low (liquidity zones)
    prev_day_high = df_1d['high'].shift(1)  # Previous day's high
    prev_day_low = df_1d['low'].shift(1)    # Previous day's low
    # Align to 1h timeframe
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high.values)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low.values)
    
    # Volume confirmation (20-period average on 1h)
    vol_ma_period = 20
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, vol_ma_period)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20) + 5  # need enough history
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(ema_4h_aligned[i]) or np.isnan(prev_day_high_aligned[i]) or \
           np.isnan(prev_day_low_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above previous day's high with volume, in 4h uptrend
            if close[i] > prev_day_high_aligned[i] and volume_confirm and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below previous day's low with volume, in 4h downtrend
            elif close[i] < prev_day_low_aligned[i] and volume_confirm and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below previous day's low or 4h trend fails
            if close[i] < prev_day_low_aligned[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above previous day's high or 4h trend fails
            if close[i] > prev_day_high_aligned[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals