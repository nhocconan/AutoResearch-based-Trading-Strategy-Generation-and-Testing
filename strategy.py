#!/usr/bin/env python3
name = "4h_Ichimoku_Tenkan_Kijun_Cross_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: close above/below 1d EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    trend_up = close > ema_1d_50_aligned
    
    # Daily volume filter: volume > 1.8x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.8 * vol_ma20_1d_aligned
    
    # Ichimoku Tenkan-sen (9) and Kijun-sen (26)
    high9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high9 + low9) / 2
    
    high26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high26 + low26) / 2
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i])):
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
            # Long: Tenkan crosses above Kijun + daily uptrend + volume filter
            if tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun + daily downtrend + volume filter
            elif tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun or daily trend down
            if tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun or daily trend up
            if tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals