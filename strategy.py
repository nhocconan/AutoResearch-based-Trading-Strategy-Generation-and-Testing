#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 1d EMA34 is rising AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND 1d EMA34 is falling AND volume > 2.0x 20-period average.
Exit when price touches the opposite Camarilla level (S3 for long, R3 for short) or EMA34 reverses.
Uses 1d HTF for EMA34 trend to filter whipsaws. Target: 75-200 total trades over 4 years (19-50/year).
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
    # We need daily OHLC, so we'll use the 1d data to compute Camarilla for each 4h bar
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)  # for exit
    camarilla_s4 = np.full(n, np.nan)  # for exit
    
    # For each 4h bar, use the most recent completed 1d bar's OHLC
    for i in range(n):
        # Find the index of the most recent completed 1d bar
        # We'll use the 1d data index that corresponds to the current time
        # Since we're using aligned arrays conceptually, we need to map 4h bar to 1d bar
        # Simpler approach: compute Camarilla once per 1d bar and forward-fill
        pass
    
    # Instead, compute Camarilla levels using the previous 1d bar's OHLC for each 4h bar
    # We'll create arrays of daily OHLC aligned to 4h bars
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Extract 1d OHLC
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3_1d = np.full(len(df_1d), np.nan)
    camarilla_s3_1d = np.full(len(df_1d), np.nan)
    camarilla_r4_1d = np.full(len(df_1d), np.nan)
    camarilla_s4_1d = np.full(len(df_1d), np.nan)
    
    for j in range(len(df_1d)):
        if j == 0:  # Need previous day's data
            continue
        # Use previous day's OHLC (j-1) to calculate levels for day j
        prev_close = close_1d[j-1]
        prev_high = high_1d[j-1]
        prev_low = low_1d[j-1]
        prev_open = open_1d[j-1]
        
        range_ = prev_high - prev_low
        camarilla_r3_1d[j] = prev_close + range_ * 1.1 / 4
        camarilla_s3_1d[j] = prev_close - range_ * 1.1 / 4
        camarilla_r4_1d[j] = prev_close + range_ * 1.1 / 2
        camarilla_s4_1d[j] = prev_close - range_ * 1.1 / 2
    
    # Align 1d Camarilla levels to 4h bars (forward fill from previous day's close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
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
                # Long exit: price touches S3 (or S4 for stronger exit) OR EMA34 starts falling
                if price < s3 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R3 (or R4 for stronger exit) OR EMA34 starts rising
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