#!/usr/bin/env python3
"""
12h_1w_Camarilla_R3_S3_Breakout_TrendFilter_Volume
Hypothesis: Uses weekly price trend (above/below 1w EMA50) to filter trades.
Enters on 12h when price breaks Camarilla R3 or S3 with volume confirmation.
Exits on opposite Camarilla level (S3 for longs, R3 for shorts).
Designed to work in both bull and bear markets by following higher-timeframe weekly trend.
Targets low trade frequency (12-37/year) via weekly trend filter and Camarilla breakout logic.
"""

name = "12h_1w_Camarilla_R3_S3_Breakout_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    # Typical price
    typical = (high + low + close) / 3
    # Range
    range_val = high - low
    
    # Camarilla levels
    R4 = close + range_val * 1.500
    R3 = close + range_val * 1.250
    R2 = close + range_val * 1.166
    R1 = close + range_val * 1.083
    PP = typical
    S1 = close - range_val * 1.083
    S2 = close - range_val * 1.166
    S3 = close - range_val * 1.250
    S4 = close - range_val * 1.500
    
    return R3, S3

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Trend Filter (EMA50) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend
    close_w = df_1w['close'].values
    ema_50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Weekly trend: 1 if price above EMA50, -1 if below
    trend_w = np.where(close_w > ema_50_w, 1, -1)
    
    # Align weekly EMA50 and trend to 12h timeframe
    ema_50_w_12h = align_htf_to_ltf(prices, df_1w, ema_50_w)
    trend_w_12h = align_htf_to_ltf(prices, df_1w, trend_w)
    
    # --- 12h Camarilla Levels (using previous day's data) ---
    # We need 1d data to calculate Camarilla levels for the current 12h bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    high_1d = df_1d['high'].values[:-1]  # Exclude today
    low_1d = df_1d['low'].values[:-1]
    close_1d = df_1d['close'].values[:-1]
    
    # Prepend NaN for the first bar (no previous day)
    high_1d = np.concatenate([[np.nan], high_1d])
    low_1d = np.concatenate([[np.nan], low_1d])
    close_1d = np.concatenate([[np.nan], close_1d])
    
    R3_1d, S3_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 12h timeframe
    R3_1d_12h = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_12h = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    # --- Volume Spike Detection (24-period average = 12 days) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_w_12h[i]) or np.isnan(trend_w_12h[i]) or 
            np.isnan(R3_1d_12h[i]) or np.isnan(S3_1d_12h[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        # Weekly trend direction
        weekly_trend = trend_w_12h[i]
        
        if position == 0:
            # Long: weekly uptrend + price breaks above R3 + volume
            if (weekly_trend == 1 and 
                close[i] > R3_1d_12h[i] and 
                close[i-1] <= R3_1d_12h[i-1] and  # crossed above this bar
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price breaks below S3 + volume
            elif (weekly_trend == -1 and 
                  close[i] < S3_1d_12h[i] and 
                  close[i-1] >= S3_1d_12h[i-1] and  # crossed below this bar
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below S3
                if (close[i] < S3_1d_12h[i] and close[i-1] >= S3_1d_12h[i-1]):  # crossed below
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above R3
                if (close[i] > R3_1d_12h[i] and close[i-1] <= R3_1d_12h[i-1]):  # crossed above
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals