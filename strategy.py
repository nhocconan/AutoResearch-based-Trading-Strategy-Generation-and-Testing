#!/usr/bin/env python3
name = "1d_WeeklyTrend_DailyPullback_Entry_v5"
timeframe = "1d"
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
    
    # ===== Weekly Trend Filter (HTF) =====
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # ===== Daily Pullback Entry =====
    ema21_daily = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    atr_raw = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    
    # ===== Daily Volume Spike Filter =====
    vol_avg_daily = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_daily = volume > (1.5 * vol_avg_daily)
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(ema21_daily[i]) or np.isnan(atr_raw[i]) or
            np.isnan(vol_spike_daily[i])):
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
        
        atr = atr_raw[i]
        
        if position == 0:
            # Long: Uptrend + pullback to EMA21 + volume spike
            if (close[i] > ema34_1w_aligned[i] and
                low[i] <= ema21_daily[i] and
                vol_spike_daily[i]):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + pullback to EMA21 + volume spike
            elif (close[i] < ema34_1w_aligned[i] and
                  high[i] >= ema21_daily[i] and
                  vol_spike_daily[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA21 or trend reversal
            if close[i] < ema21_daily[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA21 or trend reversal
            if close[i] > ema21_daily[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals