#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Trade 6h Camarilla R3/S3 breakouts with 1d EMA50 trend filter and volume spike confirmation.
In strong daily trends, Camarilla R3/S3 breakouts have high continuation probability.
Volume spike reduces false breakouts. Designed for low frequency (12-37 trades/year) to minimize fee drag.
Works in both bull and bear markets by following 1d trend.
"""

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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H+L+C)/3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla levels: R3 = CP + (H-L)*1.1/2, S3 = CP - (H-L)*1.1/2
    camarilla_pivot = typical_price.values
    camarilla_r3 = camarilla_pivot + (range_hl * 1.1 / 2).values
    camarilla_s3 = camarilla_pivot - (range_hl * 1.1 / 2).values
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1d EMA (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        
        # Volume confirmation: 1.5x average volume
        if i >= 20:
            vol_avg = np.mean(volume[max(0, i-20):i])
        else:
            vol_avg = volume_val
        
        vol_spike = volume_val > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above R3, uptrend (close > 1d EMA50), volume spike
            long_signal = (high_val > r3_val) and \
                          (close_val > ema_50_1d_val) and \
                          vol_spike
            # Short: price breaks below S3, downtrend (close < 1d EMA50), volume spike
            short_signal = (low_val < s3_val) and \
                           (close_val < ema_50_1d_val) and \
                           vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Exit: trend reversal (close < 1d EMA50) or price retraces below R3 after minimum holding
            if bars_since_entry >= 4 and ((close_val < ema_50_1d_val) or (close_val < r3_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Exit: trend reversal (close > 1d EMA50) or price retraces above S3 after minimum holding
            if bars_since_entry >= 4 and ((close_val > ema_50_1d_val) or (close_val > s3_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0