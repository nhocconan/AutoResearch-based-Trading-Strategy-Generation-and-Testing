#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm
Hypothesis: For 1h timeframe, use 4h Camarilla R1/S1 breakouts with volume confirmation and 4h trend filter.
Only take breakouts in direction of 4h trend (close > 4h EMA20 for long, close < 4h EMA20 for short).
Uses discrete sizing (0.20) to minimize fee drag. Target: 60-150 total trades over 4 years (15-37/year).
Session filter 08-20 UTC to avoid low-liquidity hours. Works in bull via breakouts, in bear via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 1h (based on previous bar's range)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    
    # Volume confirmation: volume > 1.5x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (vol_median * 1.5)
    
    # Load 4h data for HTF trend filter and Camarilla context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Start after warmup (need 20-period for volume median)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(session_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Only trade during session
        if not session_filter[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Long logic: break above R1 with volume confirmation and 4h uptrend
        long_condition = (close[i] > r1[i]) and volume_confirm[i] and (close[i] > ema_20_4h_aligned[i])
        # Short logic: break below S1 with volume confirmation and 4h downtrend
        short_condition = (close[i] < s1[i]) and volume_confirm[i] and (close[i] < ema_20_4h_aligned[i])
        
        # Exit logic: opposite Camarilla level touch
        exit_long = (close[i] < s1[i])
        exit_short = (close[i] > r1[i])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0