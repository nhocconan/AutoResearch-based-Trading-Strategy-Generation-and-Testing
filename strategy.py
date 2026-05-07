#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for Camarilla levels (HLC from previous day)
    # Camarilla levels use previous day's H, L, C
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = np.roll(close_1d, 1)
    close_1d_shift[0] = np.nan  # First value invalid
    
    # Calculate Camarilla levels for each 1d bar
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = close_1d_shift + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d_shift - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1d data for volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    # Calculate current 12h volume ratio (vs 20-period 1d average)
    # Use rolling average of 12h volume for comparison
    vol_ma10_12h = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_ratio = volume / vol_ma10_12h  # Current 12h vol vs recent 12h avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma20_aligned[i]) or
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 1d EMA34 (uptrend), breaks above Camarilla R3, volume spike
            if (close[i] > ema_34_1d_aligned[i] and 
                high[i] > camarilla_r3_aligned[i] and  # Break above R3
                volume_ratio[i] > 1.8):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA34 (downtrend), breaks below Camarilla S3, volume spike
            elif (close[i] < ema_34_1d_aligned[i] and 
                  low[i] < camarilla_s3_aligned[i] and  # Break below S3
                  volume_ratio[i] > 1.8):  # Volume spike
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Camarilla S3 (reversal) or trend change
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Camarilla R3 (reversal) or trend change
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals