#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS"
timeframe = "12h"
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
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d
    def calculate_camarilla(high_val, low_val, close_val):
        range_val = high_val - low_val
        if range_val <= 0:
            return None
        multiplier = 1.1 / 12
        S1 = close_val - range_val * multiplier * 1
        S2 = close_val - range_val * multiplier * 2
        S3 = close_val - range_val * multiplier * 3
        R1 = close_val + range_val * multiplier * 1
        R2 = close_val + range_val * multiplier * 2
        R3 = close_val + range_val * multiplier * 3
        return S1, S2, S3, R1, R2, R3
    
    # Pre-calculate daily Camarilla levels
    high_daily = df_1d['high'].values
    low_daily = df_1d['low'].values
    close_daily = df_1d['close'].values
    
    S1_daily = np.full(len(df_1d), np.nan)
    S2_daily = np.full(len(df_1d), np.nan)
    S3_daily = np.full(len(df_1d), np.nan)
    R1_daily = np.full(len(df_1d), np.nan)
    R2_daily = np.full(len(df_1d), np.nan)
    R3_daily = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        result = calculate_camarilla(high_daily[i], low_daily[i], close_daily[i])
        if result is not None:
            S1_daily[i], S2_daily[i], S3_daily[i], R1_daily[i], R2_daily[i], R3_daily[i] = result
    
    # Align Camarilla levels to 12h timeframe
    S1_12h = align_htf_to_ltf(prices, df_1d, S1_daily)
    S2_12h = align_htf_to_ltf(prices, df_1d, S2_daily)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3_daily)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1_daily)
    R2_12h = align_htf_to_ltf(prices, df_1d, R2_daily)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3_daily)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ok = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA20
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if Camarilla levels not ready
        if np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume + 1d uptrend
            if (close[i] > R3_12h[i] and 
                vol_ok[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume + 1d downtrend
            elif (close[i] < S3_12h[i] and 
                  vol_ok[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price closes below R1 or 1d trend turns down
            if (close[i] < R1_12h[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price closes above S1 or 1d trend turns up
            if (close[i] > S1_12h[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals