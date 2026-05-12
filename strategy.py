#!/usr/bin/env python3
name = "6h_WeeklyTrend_DailyPullback_Entry"
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
    
    # ===== 1d Weekly Trend Filter (HTF) =====
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    weekly_ema = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1d, weekly_ema)
    
    # ===== 6h Daily Pullback Setup =====
    # 6h EMA for pullback entry
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # ===== Daily Volume Spike Filter =====
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.8 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_ema_aligned[i]) or 
            np.isnan(ema8[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Weekly uptrend (price > weekly EMA) + 6h pullback to EMA8 + volume spike
            if (close[i] > weekly_ema_aligned[i] and
                low[i] <= ema8[i] and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend (price < weekly EMA) + 6h bounce to EMA8 + volume spike
            elif (close[i] < weekly_ema_aligned[i] and
                  high[i] >= ema8[i] and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Weekly trend reversal or 6h close above EMA8 (take profit)
            if close[i] < weekly_ema_aligned[i] or close[i] > ema8[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Weekly trend reversal or 6h close below EMA8 (take profit)
            if close[i] > weekly_ema_aligned[i] or close[i] < ema8[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals