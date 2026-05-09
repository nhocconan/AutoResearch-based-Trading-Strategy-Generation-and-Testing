#!/usr/bin/env python3
"""
6h_Ichimoku_1dTrend_Volume
Hypothesis: Use Ichimoku Cloud (Tenkan/Kijun cross + price relative to cloud) on 6h,
filtered by 1d EMA50 trend direction and volume spike.
In bull markets: price above cloud + TK cross up + 1d uptrend = long.
In bear markets: price below cloud + TK cross down + 1d downtrend = short.
Volume confirms momentum. Designed for low trade frequency (15-30/year).
"""

name = "6h_Ichimoku_1dTrend_Volume"
timeframe = "6h"
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
    
    # Ichimoku Cloud components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Cloud: between Senkou A and Senkou B
    # For cloud color: Senkou A > Senkou B = bullish cloud
    # We'll use price position relative to cloud
    # Price above cloud: close > max(senkou_a, senkou_b)
    # Price below cloud: close < min(senkou_a, senkou_b)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # TK Cross: Tenkan crosses Kijun
    tk_cross_up = (tenkan > kijun) & (tenkan <= kijun)  # Will fix with proper shift
    tk_cross_down = (tenkan < kijun) & (tenkan >= kijun)  # Will fix
    # Proper TK cross with shift
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    # Handle first element
    tk_cross_up[0] = False
    tk_cross_down[0] = False
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tk_cross_up[i]) or np.isnan(tk_cross_down[i]) or np.isnan(price_above_cloud[i]) or
            np.isnan(price_below_cloud[i]) or np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above cloud + TK cross up + 1d uptrend + volume spike + session
            if price_above_cloud[i] and tk_cross_up[i] and trend_up[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross down + 1d downtrend + volume spike + session
            elif price_below_cloud[i] and tk_cross_down[i] and trend_down[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price drops below cloud or trend reversal
            if close[i] < cloud_top[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above cloud or trend reversal
            if close[i] > cloud_bottom[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals