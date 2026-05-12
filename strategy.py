#!/usr/bin/env python3
name = "1d_WickReversal_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly volatility filter (ATR-based)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w_arr, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w_arr, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w_arr[0])
    tr3[0] = np.abs(low_1w[0] - close_1w_arr[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    
    # Daily price action: long upper shadow, short lower shadow
    body = np.abs(close - open_)
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low
    # Avoid division by zero
    body_safe = np.where(body == 0, 0.001, body)
    upper_shadow_ratio = upper_shadow / body_safe
    lower_shadow_ratio = lower_shadow / body_safe
    
    # Volume filter: volume > 1.5x 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(atr_10_1w_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: strong bullish rejection (long lower shadow) + above weekly EMA20 + volume filter
            if (lower_shadow_ratio[i] > 2.0 and 
                close[i] > ema_20_1w_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: strong bearish rejection (long upper shadow) + below weekly EMA20 + volume filter
            elif (upper_shadow_ratio[i] > 2.0 and 
                  close[i] < ema_20_1w_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly EMA20 or weak bullish signal
            if close[i] < ema_20_1w_aligned[i] or lower_shadow_ratio[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly EMA20 or weak bearish signal
            if close[i] > ema_20_1w_aligned[i] or upper_shadow_ratio[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals