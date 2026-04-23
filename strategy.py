#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume spike.
Long when price breaks above Camarilla R1 AND price > 4h EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S1 AND price < 4h EMA34 AND volume > 2.0x 20-period average.
Exit when price reverts to Camarilla pivot point OR ATR trailing stop (2.5*ATR from extreme).
Uses 4h HTF for trend alignment and 1d HTF for volume regime filter (low volume = no trades).
Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing 0.20.
Session filter: 08-20 UTC to avoid low-liquidity hours.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need enough for EMA
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1d volume average (20-period) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for MA
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Camarilla levels from previous day's OHLC (to avoid look-ahead)
    # Use daily OHLC shifted by 1 to get previous day's values
    df_1d = get_htf_data(prices, '1d')  # Reload for OHLC (small overhead, called once)
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get previous day's values for today's Camarilla calculation
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Camarilla calculation: based on previous day's range
    range_1d = high_1d_prev - low_1d_prev
    camarilla_pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    camarilla_r1 = camarilla_pivot + (range_1d * 1.1 / 12)
    camarilla_s1 = camarilla_pivot - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 1)  # vol_ma20, ema_34_4h, and +1 for roll
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_4h_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_regime = vol_ma_1d_aligned[i]  # 1d volume MA for regime filter
        atr_val = atr[i]
        
        # Volume regime filter: only trade when 1d volume is above average
        # Avoid trading in extremely low volume days
        if vol_regime <= 0 or volume[i] < 0.5 * vol_ma_val:  # Additional intraday volume filter
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND price > 4h EMA34 AND volume spike
            if price > r1 and price > ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.20
                position = 1
                highest_since_entry = price
            # Short: price breaks below S1 AND price < 4h EMA34 AND volume spike
            elif price < s1 and price < ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.20
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price reverts to pivot point
            if position == 1 and price < pivot:
                exit_signal = True
            elif position == -1 and price > pivot:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSpike_PivotExit_ATRTrailingStop_Session"
timeframe = "1h"
leverage = 1.0