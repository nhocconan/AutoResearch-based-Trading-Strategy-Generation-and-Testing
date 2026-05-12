#!/usr/bin/env python3
# 4h_Ichimoku_Cloud_TK_Cross_1dTrend
# Hypothesis: Use Ichimoku Tenkan-Kijun cross on 4h with daily EMA trend filter and volume confirmation.
# Long when Tenkan crosses above Kijun, price above Kumo (cloud), price > daily EMA, and volume > 1.5x average.
# Short when Tenkan crosses below Kijun, price below Kumo, price < daily EMA, and volume > 1.5x average.
# Exit when Tenkan crosses back in opposite direction or price crosses Kijun.
# Ichimoku provides multiple confirmation layers (trend, momentum, support/resistance) reducing false signals.
# Works in bull markets via upward cloud and in bear markets via downward cloud, filtered by daily trend.
# Targets 20-35 trades/year to minimize fee drag.

name = "4h_Ichimoku_Cloud_TK_Cross_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Kumo (cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For cloud at current period, we use Senkou A/B from 26 periods ago
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or
            np.isnan(daily_ema_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_shifted[i], senkou_b_shifted[i])
        cloud_bottom = min(senkou_a_shifted[i], senkou_b_shifted[i])
        
        if position == 0:
            # LONG: Tenkan crosses above Kijun, price above cloud, price > daily EMA, volume > 1.5x MA
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and  # cross up
                close[i] > cloud_top and 
                close[i] > daily_ema_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan crosses below Kijun, price below cloud, price < daily EMA, volume > 1.5x MA
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and  # cross down
                  close[i] < cloud_bottom and 
                  close[i] < daily_ema_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan crosses below Kijun or price drops below Kijun
            if (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]) or close[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan crosses above Kijun or price rises above Kijun
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]) or close[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals