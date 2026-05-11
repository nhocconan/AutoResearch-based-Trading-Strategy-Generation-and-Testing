#!/usr/bin/env python3
name = "1h_4h1d_Trend_Filter_PriceAction"
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
    
    # 4h trend: close above/below 4h EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    trend_up_4h = close > ema_4h_aligned
    
    # 1d trend: close above/below 1d EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up_1d = close > ema_1d_aligned
    
    # 1h price action: higher high and higher low for long, lower high and lower low for short
    # Using 2-bar lookback for pattern confirmation
    hh = (high > np.roll(high, 1)) & (high > np.roll(high, 2))
    hl = (low > np.roll(low, 1)) & (low > np.roll(low, 2))
    lh = (high < np.roll(high, 1)) & (high < np.roll(high, 2))
    ll = (low < np.roll(low, 1)) & (low < np.roll(low, 2))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
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
            # Long: Higher high and higher low + 4h uptrend + 1d uptrend + volume filter
            if hh[i] and hl[i] and trend_up_4h[i] and trend_up_1d[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: Lower high and lower low + 4h downtrend + 1d downtrend + volume filter
            elif lh[i] and ll[i] and not trend_up_4h[i] and not trend_up_1d[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Lower high and lower low or 4h trend down or 1d trend down
            if lh[i] and ll[i] or not trend_up_4h[i] or not trend_up_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Higher high and higher low or 4h trend up or 1d trend up
            if hh[i] and hl[i] or trend_up_4h[i] or trend_up_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals