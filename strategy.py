#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku Cloud with 1-week trend filter and volume confirmation.
Long when TK crosses above Kijun-Sen, price above cloud, and 1-week trend is up with volume spike.
Short when TK crosses below Kijun-Sen, price below cloud, and 1-week trend is down with volume spike.
Exit when TK crosses opposite direction or price crosses cloud.
Ichimoku provides multi-factor confirmation (momentum, trend, support/resistance) reducing false signals.
Weekly trend filter ensures alignment with higher timeframe momentum.
Volume spike confirms institutional interest.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 1-week trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 20-period EMA on 1w close for trend
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Senkou B calculation window
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom (Senkou Span A and B)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: TK crosses above Kijun, price above cloud, 1w trend up, volume spike
            tk_cross_up = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
            price_above_cloud = close[i] > cloud_top
            trend_up = ema20_1w_aligned[i] > ema20_1w_aligned[i-1]
            
            if tk_cross_up and price_above_cloud and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: TK crosses below Kijun, price below cloud, 1w trend down, volume spike
            elif (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1] and
                  close[i] < cloud_bottom and ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: TK crosses opposite direction or price crosses cloud
            exit_signal = False
            
            if position == 1:
                # Exit long: TK crosses below Kijun or price drops below cloud
                tk_cross_down = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
                price_below_cloud = close[i] < cloud_bottom
                if tk_cross_down or price_below_cloud:
                    exit_signal = True
            else:  # position == -1
                # Exit short: TK crosses above Kijun or price rises above cloud
                tk_cross_up = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
                price_above_cloud = close[i] > cloud_top
                if tk_cross_up or price_above_cloud:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0