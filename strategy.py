#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # 1d volume filter: volume > 1.8x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.8 * vol_ma20_1d_aligned
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla levels are based on previous day's range
    # We'll use the 1d OHLC to calculate levels for the current 4h period
    # Since we're on 4h timeframe, we need to shift the 1d data by 1 to get previous day
    
    # Calculate Camarilla levels for each day, then align to 4h
    # Camarilla R3, S3 levels: 
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    # But actually standard Camarilla:
    # R4 = Close + 1.5*(High - Low)
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    # S4 = Close - 1.5*(High - Low)
    
    # We need previous day's OHLC, so we shift by 1
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    # First day will have invalid data, but we'll handle with min_periods elsewhere
    
    # Calculate Camarilla levels
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + 1.1 * camarilla_range
    camarilla_s3 = prev_close_1d - 1.1 * camarilla_range
    
    # Align to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for EMA, volume, and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above Camarilla R3 + daily uptrend + volume spike
            if close[i] > camarilla_r3_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Camarilla S3 + daily downtrend + volume spike
            elif close[i] < camarilla_s3_aligned[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Camarilla S3 or daily trend down
            if close[i] < camarilla_s3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Camarilla R3 or daily trend up
            if close[i] > camarilla_r3_aligned[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals