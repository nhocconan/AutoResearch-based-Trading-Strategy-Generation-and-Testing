#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_v2
Hypothesis: On 6h timeframe, Ichimoku cloud breakouts filtered by 1d trend (price > 1d EMA50) capture medium-term momentum with fewer whipsaws. Long when price breaks above cloud in bullish 1d trend; short when price breaks below cloud in bearish 1d trend. Uses discrete sizing (±0.25) and close-based stops to target 12-37 trades/year. Works in both bull/bear markets by only trading in direction of higher-timeframe trend.
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (6h)
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
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not used for signals)
    
    # The cloud is between Senkou Span A and B
    # We use the current cloud (not shifted) for breakout detection
    # Senkou Span A and B are plotted 26 periods ahead, so to get current cloud we shift back
    senkou_a_current = np.roll(senkou_a, 26)
    senkou_b_current = np.roll(senkou_b, 26)
    # First 26 values will be invalid due to roll, but we have warmup anyway
    
    # Upper cloud boundary = max(Senkou A, Senkou B)
    # Lower cloud boundary = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a_current, senkou_b_current)
    lower_cloud = np.minimum(senkou_a_current, senkou_b_current)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Ichimoku calculations (52 for Senkou B) + 26 for cloud shift + 1d EMA50 alignment
    start_idx = max(52, 26) + 26 + 4  # +4 to ensure 1d bar completion (6h -> 1d: 4 bars per day)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        upper_cloud_val = upper_cloud[i]
        lower_cloud_val = lower_cloud[i]
        ema_50_val = ema_50_1d_aligned[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Entry conditions: price breaks above/below cloud in direction of 1d trend
        long_entry = (close_val > upper_cloud_val) and bullish_1d
        short_entry = (close_val < lower_cloud_val) and bearish_1d
        
        # Exit conditions: price returns inside cloud or trend reversal
        long_exit = (close_val < upper_cloud_val) or (close_val > lower_cloud_val) or not bullish_1d
        short_exit = (close_val > lower_cloud_val) or (close_val < upper_cloud_val) or not bearish_1d
        
        # Simplified exit: flip signal on opposite condition or cloud re-entry
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < upper_cloud_val or not bullish_1d):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > lower_cloud_val or not bearish_1d):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_v2"
timeframe = "6h"
leverage = 1.0