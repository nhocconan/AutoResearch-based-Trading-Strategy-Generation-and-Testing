#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot (R3/S3) breakout with 1w trend filter (HMA21) and volume confirmation
# Long when: price breaks above R3 AND 1w HMA21 is rising (trend up) AND volume > 2x 20-period MA
# Short when: price breaks below S3 AND 1w HMA21 is falling (trend down) AND volume > 2x 20-period MA
# Exit when: price returns to Camarilla pivot point (PP) OR trend reverses (HMA direction change)
# Uses Camarilla for structure, HMA for trend filter, volume for conviction
# Timeframe: 12h, HTF: 1w for trend. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Camarilla_R3S3_1wHMA_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for HMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need sufficient data for HMA
        return np.zeros(n)
    
    # Calculate HMA(21) on 1w
    close_1w = df_1w['close'].values
    if len(close_1w) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(data, window):
            weights = np.arange(1, window + 1)
            return np.convolve(data, weights, 'valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        hma_input = 2 * wma_half - wma_full
        hma_1w = wma(hma_input, sqrt_len)
        
        # Pad to original length
        hma_1w_padded = np.full(len(close_1w), np.nan)
        start_idx = half_len + sqrt_len - 1
        end_idx = start_idx + len(hma_1w)
        hma_1w_padded[start_idx:end_idx] = hma_1w
        
        # HMA direction: rising = 1, falling = -1
        hma_dir = np.diff(hma_1w_padded, prepend=hma_1w_padded[0])
        hma_dir = np.where(hma_dir > 0, 1, np.where(hma_dir < 0, -1, 0))
    else:
        hma_dir = np.zeros(len(df_1w))
    
    # Align 1w HMA direction to 12h timeframe
    hma_dir_aligned = align_htf_to_ltf(prices, df_1w, hma_dir.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(hma_dir_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous 12h bar
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_ = prev_high - prev_low
            
            # Camarilla levels
            pp = (prev_high + prev_low + prev_close) / 3
            r3 = prev_close + (range_ * 1.1 / 4)
            s3 = prev_close - (range_ * 1.1 / 4)
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND 1w HMA up AND volume filter
            if (close[i] > r3 and 
                hma_dir_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND 1w HMA down AND volume filter
            elif (close[i] < s3 and 
                  hma_dir_aligned[i] == -1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot point OR HMA trend turns down
            if (close[i] <= pp or hma_dir_aligned[i] == -1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot point OR HMA trend turns up
            if (close[i] >= pp or hma_dir_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals