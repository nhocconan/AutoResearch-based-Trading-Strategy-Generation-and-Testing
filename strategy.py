#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND 4h EMA20 is rising AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S1 AND 4h EMA20 is falling AND volume > 2.0x 20-period average.
Exit when price touches the opposite Camarilla level (S1 for long, R1 for short) or reverses EMA20 direction.
Uses 4h HTF for EMA20 trend to reduce whipsaws and 1d for Camarilla levels.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Session filter: 08-20 UTC to avoid low-volume periods.
Position size: 0.20 (20% of capital) to limit drawdown.
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
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA20 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d Camarilla levels (R1, S1) for entry
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla calculation: based on previous day's range
    # R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r1 = np.full(len(df_1d), np.nan)
    camarilla_s1 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):  # Start from 1 to have previous day
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        rang = prev_high - prev_low
        if rang <= 0:
            continue
        camarilla_r1[i] = prev_close + 1.1 * rang * 1.1 / 12
        camarilla_s1[i] = prev_close - 1.1 * rang * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # EMA20 (20), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_20_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_20_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA20 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_20_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Camarilla R1 AND EMA20 rising AND volume spike
            if price > r1 and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 AND EMA20 falling AND volume spike
            elif price < s1 and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S1 OR EMA20 starts falling
                if price < s1 or (i >= start_idx + 1 and ema_val < ema_20_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R1 OR EMA20 starts rising
                if price > r1 or (i >= start_idx + 1 and ema_val > ema_20_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA20_Trend_VolumeConfirmation"
timeframe = "1h"
leverage = 1.0