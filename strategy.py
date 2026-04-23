#!/usr/bin/env python3
"""
Hypothesis: 1h session-filtered 4h/1d Camarilla R3S3 breakout with volume spike.
Long when price breaks above 4h Camarilla R3 AND 1d EMA34 rising AND volume > 2x 20-period average AND UTC 08-20.
Short when price breaks below 4h Camarilla S3 AND 1d EMA34 falling AND volume > 2x 20-period average AND UTC 08-20.
Exit when price touches opposite Camarilla level (R2/S2) or EMA34 direction reverses.
Uses 4h HTF for Camarilla levels and 1d HTF for EMA34 trend filter to avoid whipsaws.
Target: 80-150 total trades over 4 years (20-37/year) with session filter reducing noise.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6
    #          S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_r3 = np.full(len(df_4h), np.nan)
    camarilla_s3 = np.full(len(df_4h), np.nan)
    camarilla_r2 = np.full(len(df_4h), np.nan)
    camarilla_s2 = np.full(len(df_4h), np.nan)
    
    for i in range(1, len(df_4h)):  # Start from 1 to use previous bar
        h = high_4h[i-1]
        l = low_4h[i-1]
        c = close_4h[i-1]
        diff = h - l
        camarilla_r3[i] = c + (diff * 1.1 / 4)
        camarilla_s3[i] = c - (diff * 1.1 / 4)
        camarilla_r2[i] = c + (diff * 1.1 / 6)
        camarilla_s2[i] = c - (diff * 1.1 / 6)
    
    # Align Camarilla levels to 1h
    r3_4h = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    r2_4h = align_htf_to_ltf(prices, df_4h, camarilla_r2)
    s2_4h = align_htf_to_ltf(prices, df_4h, camarilla_s2)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        r3 = r3_4h[i]
        s3 = s3_4h[i]
        r2 = r2_4h[i]
        s2 = s2_4h[i]
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
            # Long: Break above Camarilla R3 AND EMA34 rising AND volume spike AND in session
            if price > r3 and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S3 AND EMA34 falling AND volume spike AND in session
            elif price < s3 and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla S2 OR EMA34 starts falling
                if price < s2 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R2 OR EMA34 starts rising
                if price > r2 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0