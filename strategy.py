#!/usr/bin/env python3
name = "1d_WVWAP_Trend_1wEMA20"
timeframe = "1d"
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
    
    # 1w EMA20 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    trend_up = close > ema_1w_aligned
    
    # Daily VWAP
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, close)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for EMA, VWAP, and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above VWAP + weekly uptrend + volume filter
            if close[i] > vwap[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below VWAP + weekly downtrend + volume filter
            elif close[i] < vwap[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below VWAP or weekly trend down
            if close[i] < vwap[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above VWAP or weekly trend up
            if close[i] > vwap[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals