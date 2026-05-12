#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeFilter"
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
    
    # 12h trend: EMA50
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Daily Camarilla levels from previous day
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla levels
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])
    
    range_ = prev_high - prev_low
    camarilla_h4 = prev_close + 1.1 * range_ / 2  # H4
    camarilla_l4 = prev_close - 1.1 * range_ / 2  # L4
    camarilla_h3 = prev_close + 1.1 * range_ / 4  # H3
    camarilla_l3 = prev_close - 1.1 * range_ / 4  # L3
    
    # Align Camarilla levels (these are based on previous day's data)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure EMA has enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_12h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 0:
            # Long: 12h uptrend + price breaks above H3 + volume
            if (close[i] > ema_12h_aligned[i] and 
                close[i] > camarilla_h3_aligned[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: 12h downtrend + price breaks below L3 + volume
            elif (close[i] < ema_12h_aligned[i] and 
                  close[i] < camarilla_l3_aligned[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below L3 or trend changes
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above H3 or trend changes
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals