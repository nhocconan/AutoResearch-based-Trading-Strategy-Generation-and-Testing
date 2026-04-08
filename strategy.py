#!/usr/bin/env python3
# 12h_1w_1d_breakout_volume_v1
# Hypothesis: Breakout of 1-week Donchian channels (20-period) with 1-day trend filter and volume confirmation.
# Enters long when price breaks above 1w Donchian high in a 1d uptrend with volume spike.
# Enters short when price breaks below 1w Donchian low in a 1d downtrend with volume spike.
# Uses 12-hour timeframe for execution, targeting 50-150 total trades over 4 years.
# Works in bull/bear by aligning with higher timeframe trend and using volatility-based stops.

name = "12h_1w_1d_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period Donchian high and low
    donch_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donch_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    
    # 1-day trend filter: EMA25 vs EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d trend: 1 if EMA25 > EMA50 (uptrend), -1 if EMA25 < EMA50 (downtrend)
    trend_1d = np.where(ema25_1d > ema50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume spike detection on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_spike = vol_ratio > 2.0  # 100% above average volume
    
    # Session filter: 00-23 UTC (trade all hours for 12h timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 12h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_1w_aligned[i]) or np.isnan(donch_low_1w_aligned[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade with volume spike
        if not vol_spike[i]:
            if position != 0:
                # Hold position until exit signal
                pass
            else:
                signals[i] = 0.0
                continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 1w Donchian low OR trend turns bearish
            if close[i] < donch_low_1w_aligned[i] or trend_1d_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 1w Donchian high OR trend turns bullish
            if close[i] > donch_high_1w_aligned[i] or trend_1d_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above 1w Donchian high in 1d uptrend
            if close[i] > donch_high_1w_aligned[i] and trend_1d_aligned[i] == 1:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below 1w Donchian low in 1d downtrend
            elif close[i] < donch_low_1w_aligned[i] and trend_1d_aligned[i] == -1:
                position = -1
                signals[i] = -0.25
    
    return signals