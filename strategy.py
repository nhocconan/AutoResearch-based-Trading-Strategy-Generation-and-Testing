#!/usr/bin/env python3
"""
12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume
Hypothesis: Uses Camarilla R3/S3 levels from daily timeframe for breakout signals on 12h chart.
Requires daily EMA34 trend filter and volume confirmation. Designed to work in both bull and bear markets
by following higher-timeframe trend while using Camarilla levels for precise entries. Targets low trade frequency
(12-37/year) via tight breakout conditions and trend filter.
"""

name = "12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_ = high - low
    if range_ == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    close_val = close
    R4 = close_val + range_ * 1.1 / 2
    R3 = close_val + range_ * 1.1 / 4
    R2 = close_val + range_ * 1.1 / 6
    R1 = close_val + range_ * 1.1 / 12
    S1 = close_val - range_ * 1.1 / 12
    S2 = close_val - range_ * 1.1 / 6
    S3 = close_val - range_ * 1.1 / 4
    S4 = close_val - range_ * 1.1 / 2
    return R3, R2, R1, S1, S2, S3, S4, close_val

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla R3/S3 for Breakout Signals ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3_vals = np.full(len(close_1d), np.nan)
    s3_vals = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        R3, _, _, S1, _, S3, _, _ = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r3_vals[i] = R3
        s3_vals[i] = S3
    
    # Align daily Camarilla to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_vals)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_vals)
    
    # --- Daily EMA34 for Trend Filter ---
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Volume Spike Detection (4-period average on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 4
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema34_12h[i]) or
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
            # Long: price breaks above R3 + above EMA34 + volume
            if (close[i] > r3_12h[i] and 
                close[i] > ema34_12h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + below EMA34 + volume
            elif (close[i] < s3_12h[i] and 
                  close[i] < ema34_12h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to EMA34 or opposite Camarilla level
            if position == 1:
                # Exit long: price crosses below EMA34 OR breaks below S3
                if close[i] < ema34_12h[i] or close[i] < s3_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above EMA34 OR breaks above R3
                if close[i] > ema34_12h[i] or close[i] > r3_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals