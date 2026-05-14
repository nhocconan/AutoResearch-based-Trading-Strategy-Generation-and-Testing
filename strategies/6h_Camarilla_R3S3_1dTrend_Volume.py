#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_1dTrend_Volume
Hypothesis: On 6h timeframe, buy when price breaks above Camarilla R3 level with daily uptrend (close > EMA34) and volume confirmation; sell when breaks below S3 level with daily downtrend (close < EMA34) and volume confirmation. Uses daily EMA34 for trend filter to avoid whipsaws and volume spike for confirmation. Designed for 6h timeframe with expected 50-150 trades over 4 years to minimize fee drag while capturing trends in both bull and bear markets.
"""
name = "6h_Camarilla_R3S3_1dTrend_Volume"
timeframe = "6h"
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
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    
    # Calculate daily EMA34 for trend filter
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla levels use previous day's data
    prev_close = df_daily['close'].shift(1).values
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    
    # Calculate Camarilla levels for each day
    # R3 = C + (H-L)*1.1/2
    # S3 = C - (H-L)*1.1/2
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    
    # Volume filter: current volume > 1.5 * 50-period average volume
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + daily uptrend + volume filter
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema34_daily_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + daily downtrend + volume filter
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_daily_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Camarilla level
            if position == 1:
                if close[i] < camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals