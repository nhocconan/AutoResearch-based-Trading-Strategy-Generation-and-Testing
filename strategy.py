#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_12hCloud_Filter_Confluence
Hypothesis: 6-hour Ichimoku Tenkan-Kijun cross with 12-hour cloud filter and volume confirmation.
The Ichimoku system provides objective trend identification: price above/below cloud indicates trend direction,
TK cross provides momentum signals. Using 12h cloud as higher timeframe filter reduces false signals.
Volume confirmation ensures breakouts have participation. Designed for 6h timeframe to capture medium-term swings
while avoiding overtrading. Works in both bull and bear markets by adapting to trend via cloud filter.
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
    
    # 12h data for Ichimoku cloud (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    
    # Ichimoku components on 12h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_12h['high'].values).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_12h['low'].values).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_12h['high'].values).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_12h['low'].values).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_12h['high'].values).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_12h['low'].values).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # 6h volume confirmation: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations (52 periods for Senkou B)
    start_idx = 52 + 26  # 52 for Senkou B calculation + 26 for forward shift
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Trend filter: price relative to cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # TK cross signals
        tenkan_curr = tenkan_aligned[i]
        kijun_curr = kijun_aligned[i]
        tenkan_prev = tenkan_aligned[i-1]
        kijun_prev = kijun_aligned[i-1]
        
        # Bullish TK cross: Tenkan crosses above Kijun
        tk_bullish = (tenkan_prev <= kijun_prev) and (tenkan_curr > kijun_curr)
        # Bearish TK cross: Tenkan crosses below Kijun
        tk_bearish = (tenkan_prev >= kijun_prev) and (tenkan_curr < kijun_curr)
        
        if position == 0:
            # Look for entry signals with volume confirmation and trend alignment
            # Long: bullish TK cross + price above cloud + volume confirmation
            long_signal = tk_bullish and price_above_cloud and volume_confirmed[i]
            # Short: bearish TK cross + price below cloud + volume confirmation
            short_signal = tk_bearish and price_below_cloud and volume_confirmed[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price falls below cloud or bearish TK cross
            if curr_close < cloud_bottom or tk_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price rises above cloud or bullish TK cross
            if curr_close > cloud_top or tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_12hCloud_Filter_Confluence"
timeframe = "6h"
leverage = 1.0