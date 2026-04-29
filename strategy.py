#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1w EMA200 trend filter and volume spike
# Long when price breaks above Camarilla R3 AND price > 1w EMA200 AND volume > 2.5x 20-bar avg
# Short when price breaks below Camarilla S3 AND price < 1w EMA200 AND volume > 2.5x 20-bar avg
# Exit when price retests Camarilla pivot (central level)
# Uses 1w EMA200 for stronger trend filter to reduce false signals in choppy markets
# Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years) to avoid overtrading
# Uses stricter volume confirmation (2.5x) and higher timeframe trend filter for BTC/ETH resilience

name = "4h_Camarilla_R3S3_Breakout_1wEMA200_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get 1d data for Camarilla levels (based on previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    camarilla_r3 = prev_close_1d + camarilla_range * 1.1 / 4.0  # R3 level
    camarilla_s3 = prev_close_1d - camarilla_range * 1.1 / 4.0  # S3 level
    
    # Align Camarilla levels and pivot to 4h timeframe (they represent levels from previous 1d bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: >2.5x 20-bar average volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # volume MA and EMA200 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema200_1w = ema_200_1w_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Camarilla pivot
            if curr_close <= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Camarilla pivot
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND price > 1w EMA200 AND volume confirmation
            if curr_close > curr_r3 and curr_close > curr_ema200_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 1w EMA200 AND volume confirmation
            elif curr_close < curr_s3 and curr_close < curr_ema200_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals