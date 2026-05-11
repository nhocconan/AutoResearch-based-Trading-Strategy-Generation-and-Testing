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
    
    # Get 4h data for trend filter (EMA34)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_4h = close_4h > ema34_4h
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h)
    
    # Calculate 1h Camarilla levels based on previous day's range
    # Convert open_time to pandas datetime for date grouping
    open_time = pd.to_datetime(prices['open_time'])
    date = open_time.dt.date
    
    # Calculate previous day's high, low, close
    prev_day_high = high.copy()
    prev_day_low = low.copy()
    prev_day_close = close.copy()
    
    # Shift by 24 hours to get previous day's values (assuming hourly data)
    # For 1h timeframe, 24 bars = 1 day
    for i in range(24, n):
        if date[i] != date[i-1]:  # New day
            prev_day_high[i] = high[i-24:i].max()
            prev_day_low[i] = low[i-24:i].min()
            prev_day_close[i] = close[i-1]
        else:
            prev_day_high[i] = prev_day_high[i-1]
            prev_day_low[i] = prev_day_low[i-1]
            prev_day_close[i] = prev_day_close[i-1]
    
    # Camarilla levels for first 24 bars use previous available day
    for i in range(24):
        if i > 0:
            prev_day_high[i] = prev_day_high[i-1]
            prev_day_low[i] = prev_day_low[i-1]
            prev_day_close[i] = prev_day_close[i-1]
    
    # Calculate Camarilla levels
    range_val = prev_day_high - prev_day_low
    camarilla_r1 = prev_day_close + range_val * 1.1 / 12
    camarilla_s1 = prev_day_close - range_val * 1.1 / 12
    
    # Volume confirmation: current volume > 1.5x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > 1.5 * vol_ma24
    
    # Session filter: 08-20 UTC
    hours = open_time.dt.hour.values
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough data for EMA and calculations
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(trend_up_4h_aligned[i]) or np.isnan(vol_ma24[i])):
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
            # Long: Close above R1 + 4h uptrend + volume confirmation
            if close[i] > camarilla_r1[i] and trend_up_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: Close below S1 + 4h downtrend + volume confirmation
            elif close[i] < camarilla_s1[i] and not trend_up_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close below S1 OR 4h trend turns down
            if close[i] < camarilla_s1[i] or not trend_up_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close above R1 OR 4h trend turns up
            if close[i] > camarilla_r1[i] or trend_up_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals