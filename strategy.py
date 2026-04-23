#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
Long when price breaks above Camarilla R3 AND 1d EMA34 is rising AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND 1d EMA34 is falling AND volume > 2.0x 20-period average.
Exit when price touches the opposite Camarilla level (S3 for long, R3 for short) or reverses EMA34 direction.
Uses 1d HTF for EMA34 trend to avoid whipsaws in ranging markets. Target: 75-200 total trades over 4 years (19-50/year).
Volume spike filter (2.0x) reduces false breakouts. Discrete sizing 0.25 minimizes fee churn.
"""

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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # We need daily OHLC for 4h bars - use get_htf_data for 1d OHLC
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 1:
        return np.zeros(n)
    
    # Extract daily OHLC arrays
    daily_open = df_1d_ohlc['open'].values
    daily_high = df_1d_ohlc['high'].values
    daily_low = df_1d_ohlc['low'].values
    daily_close = df_1d_ohlc['close'].values
    
    # Calculate Camarilla levels for each daily bar
    camarilla_R3 = np.zeros(len(df_1d_ohlc))
    camarilla_S3 = np.zeros(len(df_1d_ohlc))
    
    for i in range(len(df_1d_ohlc)):
        daily_range = daily_high[i] - daily_low[i]
        camarilla_R3[i] = daily_close[i] + 1.1 * daily_range * 1.1 / 4
        camarilla_S3[i] = daily_close[i] - 1.1 * daily_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_S3)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        r3 = camarilla_R3_aligned[i]
        s3 = camarilla_S3_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA34 rising AND volume spike
            if price > r3 and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND EMA34 falling AND volume spike
            elif price < s3 and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S3 level OR EMA34 starts falling
                if price < s3 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R3 level OR EMA34 starts rising
                if price > r3 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0