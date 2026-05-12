#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Daily trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily volume spike: volume > 2.0 * 20-day avg
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Camarilla levels from previous day
    # Calculate using previous day's OHLC
    prev_day_open = np.roll(close_1d, 1)
    prev_day_high = np.roll(high_1d, 1) if 'high_1d' in locals() else np.roll(df_1d['high'].values, 1)
    prev_day_low = np.roll(low_1d, 1) if 'low_1d' in locals() else np.roll(df_1d['low'].values, 1)
    prev_day_close = np.roll(close_1d, 1)
    
    # We need high_1d and low_1d for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    prev_day_high = np.roll(high_1d, 1)
    prev_day_low = np.roll(low_1d, 1)
    
    # Camarilla R3, S3 levels
    R3 = np.zeros_like(close_1d)
    S3 = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if prev_day_high[i] == prev_day_low[i]:
            R3[i] = prev_day_close[i]
            S3[i] = prev_day_close[i]
        else:
            R3[i] = prev_day_close[i] + (prev_day_high[i] - prev_day_low[i]) * 1.1 / 4
            S3[i] = prev_day_close[i] - (prev_day_high[i] - prev_day_low[i]) * 1.1 / 4
    
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with daily uptrend and volume spike
            if (close[i] > R3_aligned[i] and
                close[i] > ema34_1d_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with daily downtrend and volume spike
            elif (close[i] < S3_aligned[i] and
                  close[i] < ema34_1d_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below S3 or loses daily uptrend
            if close[i] < S3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above R3 or loses daily downtrend
            if close[i] > R3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals