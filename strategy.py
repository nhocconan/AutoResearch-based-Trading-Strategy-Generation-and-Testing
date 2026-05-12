#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # 4h trend: EMA34
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1d volume filter: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_filter_1d = volume_1d > (1.5 * vol_avg_1d)
    vol_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_1d)
    
    # 1h Camarilla pivot (from previous hour)
    # Calculate for each hour using previous hour's data
    shift_high = pd.Series(high).shift(1).values
    shift_low = pd.Series(low).shift(1).values
    shift_close = pd.Series(close).shift(1).values
    pivot_1h = (shift_high + shift_low + shift_close) / 3.0
    r1_1h = shift_close + (shift_high - shift_low) * 1.1 / 12.0
    s1_1h = shift_close - (shift_high - shift_low) * 1.1 / 12.0
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_filter_1d_aligned[i]) or
            np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or
            np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above R1 + above 4h EMA34 + high volume day + session
            if high[i] > r1_1h[i] and close[i] > ema_34_4h_aligned[i] and vol_filter_1d_aligned[i] and session_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: breakdown below S1 + below 4h EMA34 + high volume day + session
            elif low[i] < s1_1h[i] and close[i] < ema_34_4h_aligned[i] and vol_filter_1d_aligned[i] and session_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: breakdown below S1 or below 4h EMA34
            if low[i] < s1_1h[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: breakout above R1 or above 4h EMA34
            if high[i] > r1_1h[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals