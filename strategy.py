#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1
Hypothesis: On 4h timeframe, buy when price breaks above Camarilla R3 level with daily EMA34 trend filter and volume spike; sell when breaks below S3 level with trend filter and volume spike. Uses daily EMA34 for trend direction to avoid counter-trend whipsaws and volume spike for confirmation. Camarilla levels calculated from prior day's OHLC. Designed to work in both bull and bear markets by using trend filter and mean-reversion levels.
"""
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(34, 20)  # Ensure sufficient warmup for EMA and volume
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades to reduce frequency (4h timeframe)
            if bars_since_entry < 8:
                continue
                
            # Long: price breaks above R3 + price above EMA34 + volume filter
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S3 + price below EMA34 + volume filter
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Exit: price returns to opposite Camarilla level (S3 for long, R3 for short)
            if position == 1:
                if close[i] < camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals