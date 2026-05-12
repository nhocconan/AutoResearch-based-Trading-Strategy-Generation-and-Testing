#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
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
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    camarilla_high = np.zeros(n)
    camarilla_low = np.zeros(n)
    for i in range(n):
        if i == 0:
            camarilla_high[i] = np.nan
            camarilla_low[i] = np.nan
        else:
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            camarilla_high[i] = pc + 1.1 * (ph - pl) / 2  # R3 level
            camarilla_low[i] = pc - 1.1 * (ph - pl) / 2   # S3 level
    
    # Align Camarilla levels to 6h
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_high_aligned[i]) or
            np.isnan(camarilla_low_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R3 + above 12h EMA50 + volume spike
            if close[i] > camarilla_high_aligned[i] and close[i] > ema_50_12h_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 + below 12h EMA50 + volume spike
            elif close[i] < camarilla_low_aligned[i] and close[i] < ema_50_12h_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below R3 or below 12h EMA50
            if close[i] < camarilla_high_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above S3 or above 12h EMA50
            if close[i] > camarilla_low_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals