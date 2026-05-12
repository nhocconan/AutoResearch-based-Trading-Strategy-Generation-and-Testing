#!/usr/bin/env python3
# 4h_Ichimoku_BullBear_Crossover_12hTrend_Volume
# Hypothesis: Uses Ichimoku cloud for trend direction and entry signals, filtered by 12h EMA50 trend and volume confirmation.
# Long when Tenkan crosses above Kijun above cloud with bullish 12h EMA50 trend and volume spike.
# Short when Tenkan crosses below Kijun below cloud with bearish 12h EMA50 trend and volume spike.
# Exits on opposite cross or when price exits the cloud.
# Designed for 4h timeframe with 12h trend filter to reduce false signals and work in both bull/bear markets.

name = "4h_Ichimoku_BullBear_Crossover_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2.0
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # Ensure Ichimoku components are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ema12h_trend = ema50_12h_aligned[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        vol_ratio_val = vol_ratio[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # LONG: Tenkan crosses above Kijun above cloud with bullish 12h trend and volume spike
            if (tenkan_val > kijun_val and 
                tenkan[i-1] <= kijun[i-1] and  # crossed this bar
                close[i] > cloud_top and
                close[i] > ema12h_trend and
                vol_ratio_val > 2.0):
                signals[i] = 0.30
                position = 1
            # SHORT: Tenkan crosses below Kijun below cloud with bearish 12h trend and volume spike
            elif (tenkan_val < kijun_val and 
                  tenkan[i-1] >= kijun[i-1] and  # crossed this bar
                  close[i] < cloud_bottom and
                  close[i] < ema12h_trend and
                  vol_ratio_val > 2.0):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan crosses below Kijun or price drops below cloud
            if (tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]) or close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Tenkan crosses above Kijun or price rises above cloud
            if (tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]) or close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals