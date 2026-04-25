#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter
Hypothesis: 6-hour Ichimoku Tenkan/Kijun cross with 1-day cloud filter (price above/below cloud) and volume confirmation.
Ichimoku provides trend, momentum, and support/resistance in one system. The daily cloud acts as a major trend filter -
in bull markets price stays above cloud, in bear markets below cloud. TK crosses within the cloud context provide
high-probability entries with the trend. Volume confirmation ensures breakout validity. Targets 12-30 trades/year
by requiring alignment of multiple Ichimoku components with HTF trend.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Ichimoku components (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku calculations on 1d data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df_1d['high'].rolling(window=9, min_periods=9).max().values
    period9_low = df_1d['low'].rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df_1d['high'].rolling(window=26, min_periods=26).max().values
    period26_low = df_1d['low'].rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = df_1d['high'].rolling(window=52, min_periods=52).max().values
    period52_low = df_1d['low'].rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations (52 for Senkou B) + alignment buffer
    start_idx = 52 + 26 + 1  # 52 for Senbou B calculation + 26 for shift + 1 buffer
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Cloud boundaries (Senkou Span A and B form the cloud)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Trend filter: price relative to cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # TK Cross signals
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
        
        # Previous bar TK cross (to detect fresh crosses)
        if i > start_idx:
            prev_tk_cross_up = tenkan_aligned[i-1] > kijun_aligned[i-1]
            prev_tk_cross_down = tenkan_aligned[i-1] < kijun_aligned[i-1]
            tk_cross_up_signal = tk_cross_up and not prev_tk_cross_up  # Fresh bullish cross
            tk_cross_down_signal = tk_cross_down and not prev_tk_cross_down  # Fresh bearish cross
        else:
            tk_cross_up_signal = False
            tk_cross_down_signal = False
        
        if position == 0:
            # Look for entry signals
            # Long: price above cloud + bullish TK cross + volume confirmation
            long_entry = price_above_cloud and tk_cross_up_signal and volume_confirm[i]
            # Short: price below cloud + bearish TK cross + volume confirmation
            short_entry = price_below_cloud and tk_cross_down_signal and volume_confirm[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price falls below cloud or TK cross turns bearish
            if curr_close < cloud_bottom or not tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price rises above cloud or TK cross turns bullish
            if curr_close > cloud_top or not tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter"
timeframe = "6h"
leverage = 1.0