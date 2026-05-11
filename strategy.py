#!/usr/bin/env python3
"""
12h_1w_Camarilla_R3S3_Breakout_TrendFilter_Volume
Hypothesis: Uses weekly trend filter (price above/below weekly SMA50) + Camarilla R3/S3 breakout from 1d with volume confirmation.
Designed for 12h timeframe to capture medium-term swings with low trade frequency (15-30/year).
Works in bull markets via breakouts above R3 in uptrend, and bear markets via breakdowns below S3 in downtrend.
"""

name = "12h_1w_Camarilla_R3S3_Breakout_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    range_val = high - low
    if range_val == 0:
        return np.array([close, close, close, close, close, close, close, close])
    close_last = close[-1]
    R4 = close_last + range_val * 1.1 / 2
    R3 = close_last + range_val * 1.1 / 4
    S3 = close_last - range_val * 1.1 / 4
    S4 = close_last - range_val * 1.1 / 2
    return np.array([R4, R3, S3, S4])

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly trend filter: price vs weekly SMA50 ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma50_12h = align_htf_to_ltf(prices, df_1w, weekly_sma50)
    weekly_price = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    # Weekly trend: 1 if price > SMA50, -1 if price < SMA50
    weekly_trend = np.where(weekly_price > weekly_sma50_12h, 1,
                            np.where(weekly_price < weekly_sma50_12h, -1, 0))
    
    # --- Daily Camarilla levels (R3, S3) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC to avoid look-ahead
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    camarilla = np.array([calculate_camarilla(daily_high[i], daily_low[i], daily_close[i]) 
                          for i in range(len(daily_close))])
    
    # Shift by 1 to use previous day's levels
    R3 = np.roll(camarilla[:, 1], 1)
    S3 = np.roll(camarilla[:, 2], 1)
    R3[0] = np.nan  # first day has no previous
    S3[0] = np.nan
    
    # Align to 12h
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    
    # --- Volume confirmation (24-period average = 12 days on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_trend[i]) or np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: weekly uptrend + price breaks above R3 + volume
            if (weekly_trend[i] == 1 and 
                close[i] > R3_12h[i] and 
                close[i-1] <= R3_12h[i-1] and  # crossed above this bar
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price breaks below S3 + volume
            elif (weekly_trend[i] == -1 and 
                  close[i] < S3_12h[i] and 
                  close[i-1] >= S3_12h[i-1] and  # crossed below this bar
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla level or trend flip
            if position == 1:
                # Exit long: price breaks below S3 OR weekly trend turns down
                if (close[i] < S3_12h[i] and close[i-1] >= S3_12h[i-1]) or \
                   (weekly_trend[i] == -1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R3 OR weekly trend turns up
                if (close[i] > R3_12h[i] and close[i-1] <= R3_12h[i-1]) or \
                   (weekly_trend[i] == 1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals