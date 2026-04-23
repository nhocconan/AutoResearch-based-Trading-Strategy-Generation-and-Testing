#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume confirmation.
Target: 15-37 trades/year per symbol (60-150 total over 4 years). Uses 4h/1d for signal direction, 1h only for entry timing.
Adds 08-20 UTC session filter to reduce noise. Position size fixed at 0.20 to limit drawdown.
Designed to work in both bull/bear markets via 1d trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour
    
    # Calculate 1d EMA34 for trend filter (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels (using previous 4h bar's range)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla R3, S3 levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    range_4h = high_4h - low_4h
    camarilla_r3_4h = close_4h + range_4h * 1.1 / 4
    camarilla_s3_4h = close_4h - range_4h * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need EMA34, volume MA20
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA34 = uptrend, close < 1d EMA34 = downtrend
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 1h volume > 2.0x 20-period MA (stricter to reduce trades)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND uptrend AND volume confirmation AND session
            if close[i] > camarilla_r3_aligned[i] and trend_up and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S3 AND downtrend AND volume confirmation AND session
            elif close[i] < camarilla_s3_aligned[i] and trend_down and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: reverse signal or break of opposite Camarilla level
            exit_signal = False
            if position == 1:
                # Exit long on break below Camarilla S3
                if close[i] < camarilla_s3_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Camarilla R3
                if close[i] > camarilla_r3_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_SessionFilter"
timeframe = "1h"
leverage = 1.0