#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # 1d volume filter: volume > 1.5x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.5 * vol_ma20_1d_aligned
    
    # Ichimoku Cloud (6h): Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52)
    # Tenkan-sen = (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen = (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A = (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B = (52-period high + 52-period low) / 2, shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Current cloud: Senkou Span A/B from 26 periods ago
    senkou_a_lag = np.roll(senkou_a, 26)
    senkou_b_lag = np.roll(senkou_b, 26)
    senkou_a_lag[:26] = np.nan
    senkou_b_lag[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_lag, senkou_b_lag)
    cloud_bottom = np.minimum(senkou_a_lag, senkou_b_lag)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need enough data for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
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
            # Long: Price above cloud + bullish TK cross + 1d uptrend + volume filter
            if (close[i] > cloud_top[i] and tenkan[i] > kijun[i] and 
                trend_up[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud + bearish TK cross + 1d downtrend + volume filter
            elif (close[i] < cloud_bottom[i] and tenkan[i] < kijun[i] and 
                  not trend_up[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price below cloud or bearish TK cross or 1d trend down
            if (close[i] < cloud_bottom[i] or tenkan[i] < kijun[i] or 
                not trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price above cloud or bullish TK cross or 1d trend up
            if (close[i] > cloud_top[i] or tenkan[i] > kijun[i] or 
                trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals