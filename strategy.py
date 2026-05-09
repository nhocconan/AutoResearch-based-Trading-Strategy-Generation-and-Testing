#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_ForceIndex_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA13 for trend
    ema_13_1w = pd.Series(df_1w['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_w = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Daily data for Elder Ray components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_d = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_d
    bear_power = low - ema_13_d
    
    # Force Index (13-period) = (Close - Close.prev) * Volume
    close_series = pd.Series(close)
    price_change = close_series.diff(1).fillna(0).values
    force_index_raw = price_change * volume
    force_index = pd.Series(force_index_raw).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_w[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(force_index[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: Bull Power positive, Force Index positive, weekly uptrend
            if (bull_power[i] > 0 and 
                force_index[i] > 0 and 
                close[i] > ema_13_w[i] and  # weekly uptrend
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative, Force Index negative, weekly downtrend
            elif (bear_power[i] < 0 and 
                  force_index[i] < 0 and 
                  close[i] < ema_13_w[i] and  # weekly downtrend
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative or Force Index turns negative
            if bull_power[i] <= 0 or force_index[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive or Force Index turns positive
            if bear_power[i] >= 0 or force_index[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals