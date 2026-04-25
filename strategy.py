#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend_v1
Hypothesis: 6h Ichimoku Tenkan/Kijun cross with 1d cloud filter for trend alignment.
Ichimoku provides built-in support/resistance (cloud) and momentum (TK cross).
Using 1d cloud as trend filter ensures we trade in direction of higher timeframe trend.
Targets 12-37 trades/year by requiring: 1) TK cross on 6h, 2) price relative to 1d cloud (trend filter),
3) volume > 1.5x 20-period average for confirmation.
Works in both bull/bear: cloud acts as dynamic support/resistance, TK cross captures momentum shifts.
"""

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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Ichimoku cloud (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # The cloud is between Senkou A and Senkou B
    # Top of cloud = max(Senkou A, Senkou B)
    # Bottom of cloud = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # 6h Tenkan/Kijun for TK cross (using same calculation but on 6h data)
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # TK cross signals
    # Bullish cross: Tenkan crosses above Kijun
    # Bearish cross: Tenkan crosses below Kijun
    tk_bullish = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_bearish = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations (52 for Senkou B)
    start_idx = 53
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(tenkan_6h[i]) or
            np.isnan(kijun_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation
            # Price above cloud = bullish bias, below cloud = bearish bias
            price_above_cloud = curr_close > cloud_top[i]
            price_below_cloud = curr_close < cloud_bottom[i]
            
            # Long entry: bullish TK cross + price above cloud + volume confirmation
            long_entry = tk_bullish[i] and price_above_cloud and volume_confirm[i]
            # Short entry: bearish TK cross + price below cloud + volume confirmation
            short_entry = tk_bearish[i] and price_below_cloud and volume_confirm[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price crosses below cloud bottom or bearish TK cross
            if curr_close < cloud_bottom[i] or tk_bearish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above cloud top or bullish TK cross
            if curr_close > cloud_top[i] or tk_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_v1"
timeframe = "6h"
leverage = 1.0