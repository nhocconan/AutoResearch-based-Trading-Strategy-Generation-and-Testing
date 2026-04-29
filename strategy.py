#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 Breakout with 4h EMA50 trend filter and volume spike
# Long when price breaks above R3, 4h EMA50 up-trend, volume > 2.0x average, and 08-20 UTC session
# Short when price breaks below S3, 4h EMA50 down-trend, volume > 2.0x average, and 08-20 UTC session
# Exit when price reverts to Camarilla Pivot Point
# Uses discrete position sizing (0.20) to balance capture and risk.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
# Uses 4h/1d for signal direction, 1h only for entry timing.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_v1"
timeframe = "1h"
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's OHLC for Camarilla levels
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate Camarilla levels
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    r3 = camarilla_pivot + (range_val * 1.1 / 4)
    s3 = camarilla_pivot - (range_val * 1.1 / 4)
    
    # Align 1d levels to 1h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume MA and 4h EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Camarilla Pivot
            if curr_close < curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price above Camarilla Pivot
            if curr_close > curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above R3, 4h EMA50 up-trend, volume confirmed
            if curr_close > curr_r3 and curr_close > curr_ema50_4h and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S3, 4h EMA50 down-trend, volume confirmed
            elif curr_close < curr_s3 and curr_close < curr_ema50_4h and vol_confirmed:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals