#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Camarilla pivot levels (based on previous week)
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    H = high_1w
    L = low_1w
    C = close_1w
    R3 = C + ((H - L) * 1.1 / 4)
    S3 = C - ((H - L) * 1.1 / 4)
    R4 = C + ((H - L) * 1.1 / 2)
    S4 = C - ((H - L) * 1.1 / 2)
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly data to 12h
    R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3)
    R4_1w_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S4_1w_aligned = align_htf_to_ltf(prices, df_1w, S4)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily volume spike: current volume > 2x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # ensure weekly indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_1w_aligned[i]) or np.isnan(S3_1w_aligned[i]) or 
            np.isnan(R4_1w_aligned[i]) or np.isnan(S4_1w_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + weekly uptrend + volume spike
            if (close[i] > R3_1w_aligned[i] and 
                ema34_1w_aligned[i] > ema34_1w_aligned[max(i-1, start_idx)] and  # rising trend
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + weekly downtrend + volume spike
            elif (close[i] < S3_1w_aligned[i] and 
                  ema34_1w_aligned[i] < ema34_1w_aligned[max(i-1, start_idx)] and  # falling trend
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 or weekly trend turns down
            if close[i] < S3_1w_aligned[i] or ema34_1w_aligned[i] < ema34_1w_aligned[max(i-1, start_idx)]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R3 or weekly trend turns up
            if close[i] > R3_1w_aligned[i] or ema34_1w_aligned[i] > ema34_1w_aligned[max(i-1, start_idx)]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals