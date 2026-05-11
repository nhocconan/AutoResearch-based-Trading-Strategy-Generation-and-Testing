#!/usr/bin/env python3
name = "4h_Ichimoku_Tenkan_Kijun_Cross_VolumeFilter_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Ichimoku on 4h data
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_4h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_4h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_4h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_4h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Align Ichimoku to 4h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_4h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_4h, kijun)
    
    # Volume filter: 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1d = close[i] > ema50_1d_aligned[i]
        price_below_ema1d = close[i] < ema50_1d_aligned[i]
        tenkan_above_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_below_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 0:
            # Long: Tenkan crosses above Kijun + above 1d EMA50 + volume spike
            if tenkan_above_kijun and price_above_ema1d and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun + below 1d EMA50 + volume spike
            elif tenkan_below_kijun and price_below_ema1d and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Tenkan crosses below Kijun OR trend reverses
                if tenkan_below_kijun or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: Tenkan crosses above Kijun OR trend reverses
                if tenkan_above_kijun or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals