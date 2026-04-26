#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirmation
Hypothesis: On 6h timeframe, Camarilla R3/S3 breakouts filtered by 1d trend and volume spike capture strong institutional moves with lower false breakouts than R1/S1. Long when price breaks above R3 in bullish 1d trend with volume confirmation; short when price breaks below S3 in bearish 1d trend with volume confirmation. Uses discrete sizing (±0.25) to balance risk and frequency. Works in both bull/bear markets by only trading in direction of higher-timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels (using 1d data)
    prev_high = df_1d['high'].shift(1).values  # Shift to get previous day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of calculations (20 for volume MA, 1d shift)
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > previous close, bearish if price < previous close
        bullish_1d = close_val > prev_close[i]
        bearish_1d = close_val < prev_close[i]
        
        # Entry conditions: price breaks above/below Camarilla R3/S3 levels in direction of 1d trend with volume confirmation
        long_entry = (close_val > r3_val) and bullish_1d and vol_spike
        short_entry = (close_val < s3_val) and bearish_1d and vol_spike
        
        # Exit conditions: price returns inside Camarilla levels or trend reversal
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < r3_val or not bullish_1d):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > s3_val or not bearish_1d):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0