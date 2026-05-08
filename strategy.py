#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R1 with 1d EMA50 uptrend and volume spike.
# Short when price breaks below S1 with 1d EMA50 downtrend and volume spike.
# Exit when price crosses the 50% level (C) or opposite Camarilla level.
# Uses proven Camarilla pivot structure with tight entry conditions to avoid overtrading.
# Target: 80-150 total trades over 4 years (20-38/year) for low fee drag.

name = "4h_Camarilla_R1_S1_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels using previous day's OHLC
    # We need daily OHLC for Camarilla calculation
    # Get daily data from 1d timeframe
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = close + (high - low) * 1.12
    # S1 = close - (high - low) * 1.12
    # C = close (pivot point)
    camarilla_r1 = daily_close + (daily_high - daily_low) * 1.12
    camarilla_s1 = daily_close - (daily_high - daily_low) * 1.12
    camarilla_c = daily_close  # 50% level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_1d, camarilla_c)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h volume filter: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_c_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1, 1d EMA50 uptrend, volume spike
            long_break = close[i] > camarilla_r1_aligned[i]
            long_trend = close[i] > ema50_1d_aligned[i]  # price above EMA50 = uptrend
            long_cond = long_break and long_trend and volume_filter[i]
            
            # Short conditions: price breaks below S1, 1d EMA50 downtrend, volume spike
            short_break = close[i] < camarilla_s1_aligned[i]
            short_trend = close[i] < ema50_1d_aligned[i]  # price below EMA50 = downtrend
            short_cond = short_break and short_trend and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below C (50% level) or below S1
            if close[i] < camarilla_c_aligned[i] or close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above C (50% level) or above R1
            if close[i] > camarilla_c_aligned[i] or close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals