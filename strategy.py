#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3S3_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend direction
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get daily data once for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily volume MA(20) for volume spike filter
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate daily Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    # Calculate pivot and Camarilla levels using previous day's data
    pivot = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_ = high_1d_shift - low_1d_shift
    r3 = close_1d_shift + (range_ * 1.1 / 4)
    s3 = close_1d_shift - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        ema20_4h_val = ema20_4h_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        vol_spike = volume[i] > (1.5 * vol_ma_1d_val)  # 1.5x daily volume MA
        
        if position == 0:
            if in_session:
                # Enter long: price breaks above S3 + uptrend + volume spike
                if (close[i] > s3_val and 
                    close[i] > ema20_4h_val and 
                    vol_spike):
                    signals[i] = 0.20
                    position = 1
                # Enter short: price breaks below R3 + downtrend + volume spike
                elif (close[i] < r3_val and 
                      close[i] < ema20_4h_val and 
                      vol_spike):
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR trend turns down
            if (close[i] < s3_val or close[i] < ema20_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R3 OR trend turns up
            if (close[i] > r3_val or close[i] > ema20_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals