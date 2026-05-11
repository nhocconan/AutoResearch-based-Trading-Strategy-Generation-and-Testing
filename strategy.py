#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Filter"
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
    
    # Calculate Camarilla levels using previous day's OHLC
    prev_day_high = pd.Series(high).rolling(window=24, min_periods=24).max().shift(24).values
    prev_day_low = pd.Series(low).rolling(window=24, min_periods=24).min().shift(24).values
    prev_day_close = pd.Series(close).rolling(window=24, min_periods=24).last().shift(24).values
    
    # Camarilla R1 and S1 levels
    range_ = prev_day_high - prev_day_low
    r1 = prev_day_close + (range_ * 1.1 / 12)
    s1 = prev_day_close - (range_ * 1.1 / 12)
    
    # 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_4h = close_4h > ema50_4h
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h)
    
    # Volume confirmation: current volume > 1.5x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > 1.5 * vol_ma24
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 48  # Need enough data for daily calculations
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(trend_up_4h_aligned[i]) or
            np.isnan(vol_ma24[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 + 4h uptrend + volume + session
            if close[i] > r1[i] and trend_up_4h_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 + 4h downtrend + volume + session
            elif close[i] < s1[i] and not trend_up_4h_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Price breaks below S1 OR 4h trend turns down
            if close[i] < s1[i] or not trend_up_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Price breaks above R1 OR 4h trend turns up
            if close[i] > r1[i] or trend_up_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals