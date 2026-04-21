#!/usr/bin/env python3
"""
6h_1d1w_Fibonacci_Time_Zone_Pullback
Hypothesis: Fibonacci time zones derived from major swing points on 1d chart identify potential reversal zones. 
Combine with 1w trend filter (price above/below 200 EMA) and 6s RSI pullback entries. 
Works in bull/bear by aligning with higher timeframe trend while exploiting mean-reversion at time-based zones.
Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from math import floor
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily and weekly data once
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === DAILY: Swing detection for Fibonacci time zones ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Find swing highs and lows (using 5-bar window)
    swing_high = np.zeros_like(high_1d, dtype=bool)
    swing_low = np.zeros_like(low_1d, dtype=bool)
    
    for i in range(2, len(high_1d)-2):
        if (high_1d[i] >= high_1d[i-1] and high_1d[i] >= high_1d[i-2] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            swing_high[i] = True
        if (low_1d[i] <= low_1d[i-1] and low_1d[i] <= low_1d[i-2] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            swing_low[i] = True
    
    # Get most recent significant swing (prioritize larger moves)
    last_swing_idx = -1
    last_swing_is_high = False
    max_move = 0
    
    for i in range(len(high_1d)-1, -1, -1):
        if swing_high[i]:
            move = high_1d[i] - low_1d[max(0, i-5):i+1].min()
            if move > max_move:
                max_move = move
                last_swing_idx = i
                last_swing_is_high = True
        elif swing_low[i]:
            move = high_1d[max(0, i-5):i+1].max() - low_1d[i]
            if move > max_move:
                max_move = move
                last_swing_idx = i
                last_swing_is_high = False
    
    # Calculate Fibonacci time zones from the swing point
    if last_swing_idx >= 0:
        fib_ratios = [0.382, 0.618, 1.0, 1.618, 2.618, 4.236]
        time_zones = []
        for ratio in fib_ratios:
            zone_idx = int(last_swing_idx + ratio * (len(high_1d) - last_swing_idx))
            if zone_idx < len(high_1d):
                time_zones.append(zone_idx)
        
        # Create time zone signal (1 if in zone, 0 otherwise)
        in_time_zone = np.zeros(len(high_1d), dtype=bool)
        for zone in time_zones:
            # Active for 3 days around the zone
            start = max(0, zone - 1)
            end = min(len(high_1d), zone + 2)
            in_time_zone[start:end] = True
    else:
        in_time_zone = np.zeros(len(high_1d), dtype=bool)
    
    # Align time zone signal to 6h
    time_zone_signal = align_htf_to_ltf(prices, df_1d, in_time_zone.astype(float))
    
    # === WEEKLY: Trend filter (200 EMA) ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_trend_up = align_htf_to_ltf(prices, df_1w, (close_1w > ema_200_1w).astype(float))
    
    # === 6H TIMEFRAME: RSI pullback entries ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(6) on 6s
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: above average
    vol_ma = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values  # 24*6h = 6 days
    volume_filter = volume > vol_ma * 1.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN
        if (np.isnan(time_zone_signal[i]) or np.isnan(weekly_trend_up[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        tz_signal = time_zone_signal[i]
        weekly_up = weekly_trend_up[i] > 0.5
        rsi_val = rsi[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long setup: uptrend + time zone + RSI pullback from oversold
            if weekly_up and tz_signal > 0.5 and rsi_val < 35 and vol_ok:
                # Additional: price above 6s EMA(20) for momentum
                ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
                if price > ema_20[i]:
                    signals[i] = 0.25
                    position = 1
            # Short setup: downtrend + time zone + RSI pullback from overbought
            elif not weekly_up and tz_signal > 0.5 and rsi_val > 65 and vol_ok:
                ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
                if price < ema_20[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or leaves time zone
            if rsi_val > 70 or tz_signal < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or leaves time zone
            if rsi_val < 30 or tz_signal < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d1w_Fibonacci_Time_Zone_Pullback"
timeframe = "6h"
leverage = 1.0