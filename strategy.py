#!/usr/bin/env python3
"""
1h_4h_1d_Trend_Filter_v1
Hypothesis: Use 4h Donchian channel breakout for signal direction and 1d EMA200 for trend filter on 1h timeframe.
Long when price breaks above 4h Donchian upper (20) in uptrend (close > 1d EMA200).
Short when price breaks below 4h Donchian lower (20) in downtrend (close < 1d EMA200).
Add session filter (08-20 UTC) to reduce noise trades. Fixed size 0.20 to minimize fee churn.
Target: 15-30 trades/year (60-120 total over 4 years) to stay well below fee drag threshold.
Works in both bull and bear markets by following the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 4h Donchian (20) and 1d EMA (200)
    start_idx = max(20, 200)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(donchian_high_4h_aligned[i]) or 
            np.isnan(donchian_low_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        dh_4h = donchian_high_4h_aligned[i]
        dl_4h = donchian_low_4h_aligned[i]
        ema_200_1d_val = ema_200_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian high in uptrend
            long_signal = (high_val > dh_4h) and (close_val > ema_200_1d_val)
            # Short: price breaks below 4h Donchian low in downtrend
            short_signal = (low_val < dl_4h) and (close_val < ema_200_1d_val)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price closes below 4h Donchian low or trend reversal
            if close_val < dl_4h or close_val < ema_200_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price closes above 4h Donchian high or trend reversal
            if close_val > dh_4h or close_val > ema_200_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_4h_1d_Trend_Filter_v1"
timeframe = "1h"
leverage = 1.0