#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily close for EMA trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day's range)
    # R4 = Close + (High - Low) * 1.500
    # R3 = Close + (High - Low) * 1.250
    # R2 = Close + (High - Low) * 1.166
    # R1 = Close + (High - Low) * 1.083
    # PP = (High + Low + Close) / 3
    # S1 = Close - (High - Low) * 1.083
    # S2 = Close - (High - Low) * 1.166
    # S3 = Close - (High - Low) * 1.250
    # S4 = Close - (High - Low) * 1.500
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_range_1d = np.roll(daily_range, 1)
    
    # Set first day's values to 0 (will be handled by min_periods logic)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_range_1d[0] = 0
    
    # Calculate Camarilla levels using previous day's data
    r4_1d = prev_close_1d + prev_range_1d * 1.500
    r3_1d = prev_close_1d + prev_range_1d * 1.250
    r2_1d = prev_close_1d + prev_range_1d * 1.166
    r1_1d = prev_close_1d + prev_range_1d * 1.083
    pp_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    s1_1d = prev_close_1d - prev_range_1d * 1.083
    s2_1d = prev_close_1d - prev_range_1d * 1.166
    s3_1d = prev_close_1d - prev_range_1d * 1.250
    s4_1d = prev_close_1d - prev_range_1d * 1.500
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation and uptrend
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume confirmation and downtrend
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below R3 or trend changes
            if close[i] < r3_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above S3 or trend changes
            if close[i] > s3_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals