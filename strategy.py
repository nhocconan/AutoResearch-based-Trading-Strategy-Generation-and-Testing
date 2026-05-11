#!/usr/bin/env python3
name = "6h_WeeklyPivot_Trend_ETF"
timeframe = "6h"
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
    
    # 1d EMA34 for trend filter (long-term bias)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Weekly high/low for pivot levels (simplified: weekly range)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    # Use previous week's high/low as pivot reference
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high, additional_delay_bars=1)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low, additional_delay_bars=1)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Long conditions: price above weekly high + 1d uptrend + volume spike
        if close[i] > weekly_high_aligned[i] and ema_1d_aligned[i] > ema_1d[max(0, i-1)] and volume_filter[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short conditions: price below weekly low + 1d downtrend + volume spike
        elif close[i] < weekly_low_aligned[i] and ema_1d_aligned[i] < ema_1d[max(0, i-1)] and volume_filter[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price crosses back into weekly range or trend reversal
        elif position == 1 and (close[i] < weekly_low_aligned[i] or ema_1d_aligned[i] < ema_1d[max(0, i-1)]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > weekly_high_aligned[i] or ema_1d_aligned[i] > ema_1d[max(0, i-1)]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals