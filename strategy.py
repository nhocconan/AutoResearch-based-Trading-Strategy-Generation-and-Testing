#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 1d EMA34 is rising AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND 1d EMA34 is falling AND volume > 2.0x 20-period average.
Exit when price touches opposite Camarilla level (S3 for long, R3 for short) or reverses EMA34 direction.
Uses 1d HTF for EMA34 trend filter to avoid whipsaws in ranging markets. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Calculate 12h Camarilla levels (based on previous 12h bar's OHLC)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # We use the previous completed 12h bar to calculate levels for current bar
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)  # for stop loss
    camarilla_s4 = np.full(n, np.nan)  # for stop loss
    
    # Need to resample to 12h to get proper OHLC for Camarilla calculation
    # But we must use mtf_data to avoid look-ahead and resampling issues
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_r3_12h = np.full(len(df_12h), np.nan)
    camarilla_s3_12h = np.full(len(df_12h), np.nan)
    camarilla_r4_12h = np.full(len(df_12h), np.nan)
    camarilla_s4_12h = np.full(len(df_12h), np.nan)
    
    for i in range(len(df_12h)):
        h = df_12h['high'].iloc[i]
        l = df_12h['low'].iloc[i]
        c = df_12h['close'].iloc[i]
        range_hl = h - l
        camarilla_r3_12h[i] = c + 1.1 * range_hl * 1.1 / 4
        camarilla_s3_12h[i] = c - 1.1 * range_hl * 1.1 / 4
        camarilla_r4_12h[i] = c + 1.1 * range_hl * 1.1 / 2
        camarilla_s4_12h[i] = c - 1.1 * range_hl * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (already aligned by get_htf_data)
    # Now we need to forward-fill to 15m timeframe (assuming 15m prices for now)
    # But since we don't know the timeframe, we'll use index-based alignment
    # Actually, we should align to the primary timeframe using align_htf_to_ltf
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4_12h)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4_12h)
    
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
                # Long exit: price touches S3 (or S4 for stop) OR EMA34 starts falling
                if price <= s3 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R3 (or R4 for stop) OR EMA34 starts rising
                if price >= r3 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0