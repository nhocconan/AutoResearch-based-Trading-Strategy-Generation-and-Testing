#!/usr/bin/env python3
name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
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
    
    # Camarilla levels from previous day
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla levels
    range_hl = prev_high - prev_low
    R3 = prev_close + (range_hl * 1.1 / 4)
    S3 = prev_close - (range_hl * 1.1 / 4)
    
    # Weekly trend (HMA)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # HMA: WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, n):
        if n < 1:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights/weights.sum(), mode='same')
    
    def hma(arr, n):
        if n < 1:
            return np.full_like(arr, np.nan)
        half_n = n // 2
        sqrt_n = int(np.sqrt(n))
        wma_half = wma(arr, half_n)
        wma_full = wma(arr, n)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, sqrt_n)
    
    hma_21_1w = hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Daily volume spike (20-period average)
    vol_avg = np.zeros(n)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2.0 * vol_avg)
    vol_spike = vol_spike.astype(float)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(hma_21_1w_aligned[i]) or
            np.isnan(vol_avg[i])):
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
            # Long: Price breaks above R3 + above weekly HMA21 + volume spike
            if (close[i] > R3[i] and
                close[i] > hma_21_1w_aligned[i] and
                vol_spike[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + below weekly HMA21 + volume spike
            elif (close[i] < S3[i] and
                  close[i] < hma_21_1w_aligned[i] and
                  vol_spike[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below S3 or below weekly HMA21
            if close[i] < S3[i] or close[i] < hma_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above R3 or above weekly HMA21
            if close[i] > R3[i] or close[i] > hma_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals