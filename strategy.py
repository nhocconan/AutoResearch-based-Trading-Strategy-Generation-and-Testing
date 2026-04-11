#!/usr/bin/env python3
# 1h_4h_1d_camarilla_pivot_volume_v1
# Strategy: 1h Camarilla pivot breakout with volume confirmation and 4h/1d trend filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Camarilla levels act as support/resistance; breakouts with volume
# and aligned 4h/1d trend yield high-probability trades. Works in both bull/bear
# by using trend filter to avoid counter-trend trades. Target 15-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_pivot_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(100) for trend filter
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Calculate Camarilla pivots using previous day's OHLC
    # We'll use daily OHLC from 1d data and align to 1h
    prev_day_high = df_1d['high'].shift(1).values  # Previous day high
    prev_day_low = df_1d['low'].shift(1).values    # Previous day low
    prev_day_close = df_1d['close'].shift(1).values # Previous day close
    
    # Align previous day's OHLC to 1h timeframe
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
    prev_day_close_aligned = align_htf_to_ltf(prices, df_1d, prev_day_close)
    
    # Calculate Camarilla levels for each 1h bar
    # H4 = Close + 1.5*(High-Low) - resistance level
    # L4 = Close - 1.5*(High-Low) - support level
    camarilla_high = prev_day_close_aligned + 1.5 * (prev_day_high_aligned - prev_day_low_aligned)
    camarilla_low = prev_day_close_aligned - 1.5 * (prev_day_high_aligned - prev_day_low_aligned)
    
    # Volume confirmation: current volume > 1.5x 24-period average
    vol_series = pd.Series(volume)
    vol_avg_24 = vol_series.rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > (1.5 * vol_avg_24)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend filters
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        uptrend_1d = close[i] > ema_100_1d_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        downtrend_1d = close[i] < ema_100_1d_aligned[i]
        
        uptrend = uptrend_4h and uptrend_1d
        downtrend = downtrend_4h and downtrend_1d
        
        # Entry logic: Camarilla breakout + volume + trend alignment + session
        if (close[i] > camarilla_high[i] and vol_confirm[i] and uptrend and 
            session_filter[i] and position != 1):
            position = 1
            signals[i] = 0.20
        elif (close[i] < camarilla_low[i] and vol_confirm[i] and downtrend and 
              session_filter[i] and position != -1):
            position = -1
            signals[i] = -0.20
        # Exit: price returns to pivot level or trend change
        elif position == 1 and (close[i] < prev_day_close_aligned[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > prev_day_close_aligned[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals