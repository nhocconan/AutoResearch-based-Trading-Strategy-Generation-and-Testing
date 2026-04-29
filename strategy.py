#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Long when price breaks above Camarilla R3, 12h EMA50 uptrend, volume > 2.0x average
# Short when price breaks below Camarilla S3, 12h EMA50 downtrend, volume > 2.0x average
# Exit when price reverts to Camarilla pivot point (mean reversion)
# Uses discrete position sizing (0.25) to balance capture and risk.
# Target: 100-180 total trades over 4 years (25-45/year) to avoid fee drag.
# Uses 12h for signal direction/trend, 4h only for entry timing and breakout levels.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
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
    
    # Get 4h data for Camarilla pivot calculation (using previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough data for daily pivot calculation
        return np.zeros(n)
    
    # Resample 4h to daily OHLC for Camarilla calculation (using actual daily bars would be better but we'll approximate)
    # Instead, we'll use 12h data to get proper daily OHLC
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC from 12h data (2x 12h = 1 day)
    # We'll use the last completed 24h period for pivot calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate daily OHLC: every 2 bars of 12h = 1 day
    # High of day = max of last 2 highs, Low of day = min of last 2 lows, Close of day = last close
    daily_high = np.maximum(high_12h[:-1:2], high_12h[1::2]) if len(high_12h) >= 2 else np.array([])
    daily_low = np.minimum(low_12h[:-1:2], low_12h[1::2]) if len(low_12h) >= 2 else np.array([])
    daily_close = close_12h[1::2] if len(close_12h) >= 2 else np.array([])
    
    if len(daily_high) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    camarilla_pivot = (daily_high + daily_low + daily_close) / 3
    camarilla_range = daily_high - daily_low
    camarilla_R3 = camarilla_pivot + (camarilla_range * 1.1 / 4)
    camarilla_S3 = camarilla_pivot - (camarilla_range * 1.1 / 4)
    camarilla_R4 = camarilla_pivot + (camarilla_range * 1.1 / 2)
    camarilla_S4 = camarilla_pivot - (camarilla_range * 1.1 / 2)
    
    # We need to align these daily levels to the 12h timeframe first, then to 4h
    # Each Camarilla level applies to the 24h period following its calculation
    camarilla_pivot_12h = np.repeat(camarilla_pivot, 2)  # Each day's pivot for 2x 12h bars
    camarilla_R3_12h = np.repeat(camarilla_R3, 2)
    camarilla_S3_12h = np.repeat(camarilla_S3, 2)
    camarilla_R4_12h = np.repeat(camarilla_R4, 2)
    camarilla_S4_12h = np.repeat(camarilla_S4, 2)
    
    # Handle edge cases for odd length
    if len(camarilla_pivot_12h) < len(df_12h):
        # Pad with last value if needed
        padding = len(df_12h) - len(camarilla_pivot_12h)
        if padding > 0:
            camarilla_pivot_12h = np.pad(camarilla_pivot_12h, (0, padding), 'edge')
            camarilla_R3_12h = np.pad(camarilla_R3_12h, (0, padding), 'edge')
            camarilla_S3_12h = np.pad(camarilla_S3_12h, (0, padding), 'edge')
            camarilla_R4_12h = np.pad(camarilla_R4_12h, (0, padding), 'edge')
            camarilla_S4_12h = np.pad(camarilla_S4_12h, (0, padding), 'edge')
        elif padding < 0:
            camarilla_pivot_12h = camarilla_pivot_12h[:len(df_12h)]
            camarilla_R3_12h = camarilla_R3_12h[:len(df_12h)]
            camarilla_S3_12h = camarilla_S3_12h[:len(df_12h)]
            camarilla_R4_12h = camarilla_R4_12h[:len(df_12h)]
            camarilla_S4_12h = camarilla_S4_12h[:len(df_12h)]
    
    # Align Camarilla levels from 12h to 4h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot_12h)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R3_12h)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S3_12h)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R4_12h)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S4_12h)
    
    # Get 12h data for EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 12h EMA50 and 4h volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_R3 = camarilla_R3_aligned[i]
        curr_S3 = camarilla_S3_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_R4 = camarilla_R4_aligned[i]
        curr_S4 = camarilla_S4_aligned[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Camarilla pivot point (mean reversion to pivot)
            if curr_close < curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Camarilla pivot point (mean reversion to pivot)
            if curr_close > curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average (strict filter)
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above Camarilla R3, 12h EMA50 uptrend, volume confirmed
            if curr_high > curr_R3 and curr_close > curr_ema50_12h and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3, 12h EMA50 downtrend, volume confirmed
            elif curr_low < curr_S3 and curr_close < curr_ema50_12h and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals