#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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
    volume = prices['volume'].values
    
    # === 4H DATA FOR TREND FILTER ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4H EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 1D DATA FOR VOLUME CONFIRMATION ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    # 1D volume average (20-period)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d, additional_delay_bars=0)
    
    # === 1H CAMARILLA PIVOT POINTS (based on previous day) ===
    # Calculate daily pivot from previous day's OHLC
    # We need to get the previous day's data for each 1h bar
    # Since we're using 1h timeframe, we'll use the daily data aligned to 1h
    df_1d_ohlc = get_htf_data(prices, '1d')
    # For each 1h bar, we need the previous day's OHLC
    # We'll shift the daily data by 1 to get previous day's values
    open_1d = df_1d_ohlc['open'].values
    high_1d = df_1d_ohlc['high'].values
    low_1d = df_1d_ohlc['low'].values
    close_1d = df_1d_ohlc['close'].values
    
    # Shift by 1 to get previous day's values for today's calculation
    prev_open_1d = np.roll(open_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # Set first value to NaN (no previous day)
    prev_open_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla levels for each day, then align to 1h
    # Camarilla R1 = Close + (High - Low) * 1.1/12
    # Camarilla S1 = Close - (High - Low) * 1.1/12
    camarilla_r1_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    camarilla_s1_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12
    
    # Align to 1h timeframe
    camarilla_r1_1h = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_r1_1d)
    camarilla_s1_1h = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_s1_1d)
    
    # === SESSION FILTER (08-20 UTC) ===
    # Pre-compute hours for efficiency
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure we have enough data for all indicators
    
    for i in range(start_idx, n):
        # Check if we have valid data
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(camarilla_r1_1h[i]) or 
            np.isnan(camarilla_s1_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1, above 4h EMA50 (uptrend), with volume spike
            if (close[i] > camarilla_r1_1h[i] and 
                close[i] > ema50_4h_aligned[i] and 
                volume[i] > vol_avg_1d_aligned[i] * 2.0):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S1, below 4h EMA50 (downtrend), with volume spike
            elif (close[i] < camarilla_s1_1h[i] and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume[i] > vol_avg_1d_aligned[i] * 2.0):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 OR below 4h EMA50
            if (close[i] < camarilla_s1_1h[i]) or (close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 OR above 4h EMA50
            if (close[i] > camarilla_r1_1h[i]) or (close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals