#!/usr/bin/env python3
# 1d_1w_ichimoku_trend_follow_v1
# Hypothesis: 1d Ichimoku Kumo breakout with weekly trend filter (EMA50) to avoid counter-trend trades.
# Long when price breaks above Kumo and weekly EMA50 rising; short when price breaks below Kumo and weekly EMA50 falling.
# Uses daily timeframe to target 15-35 trades/year, minimizing fee drag. Works in bull/bear via multi-timeframe alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ichimoku_trend_follow_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen: (9-period high + low) / 2
    period9_high = np.full(n, np.nan)
    period9_low = np.full(n, np.nan)
    for i in range(9, n):
        period9_high[i] = np.max(high[i-9:i+1])
        period9_low[i] = np.min(low[i-9:i+1])
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen: (26-period high + low) / 2
    period26_high = np.full(n, np.nan)
    period26_low = np.full(n, np.nan)
    for i in range(26, n):
        period26_high[i] = np.max(high[i-26:i+1])
        period26_low[i] = np.min(low[i-26:i+1])
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan + kijun) / 2
    
    # Senkou Span B: (52-period high + low) / 2
    period52_high = np.full(n, np.nan)
    period52_low = np.full(n, np.nan)
    for i in range(52, n):
        period52_high[i] = np.max(high[i-52:i+1])
        period52_low[i] = np.min(low[i-52:i+1])
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 52  # Ensure Ichimoku is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a[i], senkou_span_b[i])
        lower_cloud = np.minimum(senkou_span_a[i], senkou_span_b[i])
        
        # 1w trend filter: EMA50 slope (rising/falling)
        ema_rising = ema50_1w_aligned[i] > ema50_1w_aligned[i-1]
        ema_falling = ema50_1w_aligned[i] < ema50_1w_aligned[i-1]
        
        if position == 1:  # Long position
            # Exit: price drops below cloud or weekly trend turns bearish
            if close[i] < lower_cloud or not ema_rising:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud or weekly trend turns bullish
            if close[i] > upper_cloud or not ema_falling:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above cloud and weekly trend bullish
            if close[i] > upper_cloud and ema_rising:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below cloud and weekly trend bearish
            elif close[i] < lower_cloud and ema_falling:
                position = -1
                signals[i] = -0.25
    
    return signals