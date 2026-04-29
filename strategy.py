#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 Breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above R1, 1d EMA34 up-trend, volume > 1.8x average
# Short when price breaks below S3, 1d EMA34 down-trend, volume > 1.8x average
# Exit when price reverts to Camarilla Pivot Point
# Uses discrete position sizing (0.25) to balance capture and risk.
# Target: 80-180 total trades over 4 years (20-45/year) to avoid fee drag.
# Uses 1d for signal direction and levels, 4h only for execution timing.

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Volume_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate previous day's OHLC for Camarilla levels
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate Camarilla levels
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    r1 = camarilla_pivot + (range_val * 1.1 / 12)
    s1 = camarilla_pivot - (range_val * 1.1 / 12)
    r3 = camarilla_pivot + (range_val * 1.1 / 4)
    s3 = camarilla_pivot - (range_val * 1.1 / 4)
    
    # Align 1d levels to 4h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Volume MA and 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Camarilla Pivot
            if curr_close < curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Camarilla Pivot
            if curr_close > curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirmed = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above R1, 1d EMA34 up-trend, volume confirmed
            if curr_close > curr_r1 and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3, 1d EMA34 down-trend, volume confirmed
            elif curr_close < curr_s3 and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals