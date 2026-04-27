#!/usr/bin/env python3
"""
#100914 - 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
Hypothesis: 1h strategy using 4h EMA for trend direction, 1d volume spike for confirmation, and Camarilla breakout for entry.
Trades only during 08-20 UTC to avoid low-liquidity hours. Targets 15-37 trades/year by requiring 4h trend alignment + 1d volume spike + Camarilla breakout.
Works in bull (breakouts with trend) and bear (mean reversion to pivot after overextension).
"""

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
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA34 for trend
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 1d volume average for spike detection
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Previous day's Camarilla levels (to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    daily_pivot = (high_1d + low_1d + close_1d) / 3
    daily_range = high_1d - low_1d
    daily_r1 = close_1d + daily_range * 1.1 / 12
    daily_s1 = close_1d - daily_range * 1.1 / 12
    
    # Align Camarilla levels to 1h (using previous day's levels)
    camarilla_r1 = align_htf_to_ltf(prices, df_1d, daily_r1)
    camarilla_s1 = align_htf_to_ltf(prices, df_1d, daily_s1)
    camarilla_pivot = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside session or missing data
        if not session_mask[i] or \
           np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or \
           np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(camarilla_pivot[i]):
            signals[i] = 0.0
            continue
        
        # Volume spike: current 1h volume > 2x 1d average volume (scaled to 1h)
        # 1d volume represents 24h, so 1h average is vol_ma_1d / 24
        vol_spike = volume[i] > (2.0 * vol_ma_1d_aligned[i] / 24.0)
        
        # Long: price breaks above R1, above 4h EMA34, volume spike
        if (close[i] > camarilla_r1[i] and 
            close[i] > ema34_4h_aligned[i] and 
            vol_spike):
            signals[i] = 0.20
            position = 1
        # Short: price breaks below S1, below 4h EMA34, volume spike
        elif (close[i] < camarilla_s1[i] and 
              close[i] < ema34_4h_aligned[i] and 
              vol_spike):
            signals[i] = -0.20
            position = -1
        # Exit: price returns to Camarilla Pivot (mean reversion)
        elif position == 1 and close[i] < camarilla_pivot[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > camarilla_pivot[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0