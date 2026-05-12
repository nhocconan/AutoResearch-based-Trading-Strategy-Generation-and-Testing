#!/usr/bin/env python3
name = "6h_ElderRay_RollingRet_Trend"
timeframe = "6h"
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
    
    # ===== 6h Elder Ray (LTF) =====
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    # Smooth powers to reduce noise
    bull_smooth = pd.Series(bull_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    bear_smooth = pd.Series(bear_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # ===== 6h Rolling Return Trend Filter (LTF) =====
    ret_6 = np.zeros(n)
    ret_6[6:] = (close[6:] - close[:-6]) / close[:-6]
    ret_avg = pd.Series(ret_6).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # ===== Weekly Trend Filter (HTF) =====
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # ===== Daily Volume Spike Filter (HTF) =====
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.8 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # after EMA13, ret_6 warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_smooth[i]) or 
            np.isnan(bear_smooth[i]) or
            np.isnan(ret_avg[i]) or
            np.isnan(ema34_1w_aligned[i]) or
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
            # Long: Bull power positive AND rising, rolling return positive, above weekly EMA34, volume spike
            if (bull_smooth[i] > 0 and bull_smooth[i] > bull_smooth[i-1] and
                ret_avg[i] > 0 and
                close[i] > ema34_1w_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Bear power negative AND falling, rolling return negative, below weekly EMA34, volume spike
            elif (bear_smooth[i] < 0 and bear_smooth[i] < bear_smooth[i-1] and
                  ret_avg[i] < 0 and
                  close[i] < ema34_1w_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull power turns negative or rolling return turns negative
            if bull_smooth[i] <= 0 or ret_avg[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear power turns positive or rolling return turns positive
            if bear_smooth[i] >= 0 or ret_avg[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals