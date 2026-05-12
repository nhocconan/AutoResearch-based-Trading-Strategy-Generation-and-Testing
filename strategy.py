#!/usr/bin/env python3
name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 500:  # Need sufficient data for weekly calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter (needs one extra week for confirmation)
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_1d = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # === 1D DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    range_hl = prev_high - prev_low
    
    # R3 and S3 levels (more conservative than R4/S4)
    r3 = prev_close + range_hl * 1.1 * 1.125/1.1  # R3 = C + (H-L)*1.125
    s3 = prev_close - range_hl * 1.1 * 1.125/1.1  # S3 = C - (H-L)*1.125
    
    # Align Camarilla levels to 1d timeframe (already aligned, but use for consistency)
    r3_1d = align_htf_to_ltf(prices, df_1d, r3)
    s3_1d = align_htf_to_ltf(prices, df_1d, s3)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or 
            np.isnan(ema50_1w_1d[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume, weekly trend up
            if (close[i] > r3_1d[i] and 
                close[i] > ema50_1w_1d[i] and  # Uptrend filter
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume, weekly trend down
            elif (close[i] < s3_1d[i] and 
                  close[i] < ema50_1w_1d[i] and  # Downtrend filter
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Weekly trend breaks down
            if close[i] < ema50_1w_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly trend breaks up
            if close[i] > ema50_1w_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals