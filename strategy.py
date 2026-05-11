#!/usr/bin/env python3
name = "1h_TrendFollow_4hEMA1dTrend"
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
    
    # Get 4h data for EMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    trend_up_4h = close_4h > ema21_4h
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # 1h EMA21 for entry timing
    ema21_1h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume filter: current volume > 1.2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.2 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema21_1h[i]) or np.isnan(trend_up_4h_aligned[i]) or 
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > 1h EMA21 + 4h uptrend + 1d uptrend + volume + session
            if (close[i] > ema21_1h[i] and trend_up_4h_aligned[i] and 
                trend_up_1d_aligned[i] and volume_filter[i] and in_session[i]):
                signals[i] = 0.20
                position = 1
        elif position == 1:
            # Long exit: price < 1h EMA21 OR 4h trend down OR 1d trend down
            if (close[i] < ema21_1h[i] or not trend_up_4h_aligned[i] or 
                not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
    
    return signals