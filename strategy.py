#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_v1
Hypothesis: Trade daily Camarilla R1/S1 breakouts with weekly EMA34 trend filter and volume confirmation.
Long when price breaks above R1/S1 AND 1w close > EMA(34) AND volume spike.
Short when price breaks below S1/R1 AND 1w close < EMA(34) AND volume spike.
Targets 30-100 total trades over 4 years (7-25/year). Works in bull (breakouts continue) and bear (breakdowns continue) via 1w trend filter.
"""

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
    
    # Get 1w data for EMA trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous 1w bar
    # Using R1/S1 for breakout entries (tighter than R4/S4)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Where C = (H+L+Close)/3 of previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1w['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1w['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1w['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    
    # Align Camarilla levels to 1d
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1w EMA(34), volume MA(20), and need 1w data
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        trend_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirm AND 1w uptrend
            long_signal = (close_val > r1_aligned[i]) and vol_conf and trend_up
            
            # Short: price breaks below S1 AND volume confirm AND 1w downtrend
            short_signal = (close_val < s1_aligned[i]) and vol_conf and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price drops below S1 (failed breakout) OR 1w trend flips down
            if (close_val < s1_aligned[i]) or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 (failed breakdown) OR 1w trend flips up
            if (close_val > r1_aligned[i]) or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_v1"
timeframe = "1d"
leverage = 1.0