#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # 4h trend: close above/below 4h EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    trend_up = close > ema_4h_aligned
    
    # Daily volume filter: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.5 * vol_ma20_1d_aligned
    
    # Camarilla pivot levels (using previous day's range)
    # Calculate pivot and levels from previous day's OHLC
    prev_day_close = np.roll(close, 24)  # Previous day's close (24 hours)
    prev_day_high = np.roll(high, 24)    # Previous day's high
    prev_day_low = np.roll(low, 24)      # Previous day's low
    
    # Handle initial values
    prev_day_close[:24] = prev_day_close[24] if len(prev_day_close) > 24 else close[0]
    prev_day_high[:24] = prev_day_high[24] if len(prev_day_high) > 24 else high[0]
    prev_day_low[:24] = prev_day_low[24] if len(prev_day_low) > 24 else low[0]
    
    # Calculate Camarilla levels
    range_prev = prev_day_high - prev_day_low
    camarilla_r1 = prev_day_close + range_prev * 1.1 / 12
    camarilla_s1 = prev_day_close - range_prev * 1.1 / 12
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above Camarilla R1 + 4h uptrend + volume filter
            if close[i] > camarilla_r1[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: Close below Camarilla S1 + 4h downtrend + volume filter
            elif close[i] < camarilla_s1[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close below Camarilla S1 or 4h trend down
            if close[i] < camarilla_s1[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close above Camarilla R1 or 4h trend up
            if close[i] > camarilla_r1[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals