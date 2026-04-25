#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout with 1d Trend Filter
Hypothesis: Ichimoku cloud (Senkou Span A/B) acts as dynamic support/resistance. 
Tenkan-Kijun cross above/below cloud with 1d EMA50 trend alignment captures strong momentum moves. 
Cloud acts as natural stop: exit when price re-enters cloud. Works in both bull/bear markets 
by only taking trades aligned with higher timeframe trend. Target: 12-37 trades/year on 6h.
"""

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
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h data
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Leading Span A (Senkou Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou + low_senkou) / 2)
    
    # For cloud calculation, we need to shift Senkou Spans forward by 26 periods
    # But to avoid look-ahead, we use the values that were available 26 periods ago
    # So current cloud is based on Senkou A/B calculated 26 periods ago
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid (rolled from end)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all calculations + 26 for cloud shift
    start_idx = max(period_kijun, period_senkou_b) + 26
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        ema_trend = ema_50_1d_aligned[i]
        
        # Cloud boundaries: higher Senkou Span is resistance, lower is support
        cloud_top = max(senkou_a_shifted[i], senkou_b_shifted[i])
        cloud_bottom = min(senkou_a_shifted[i], senkou_b_shifted[i])
        
        # Tenkan-Kijun cross
        tenkan_kijun_cross = tenkan[i] > kijun[i]
        tenkan_kijun_cross_down = tenkan[i] < kijun[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Tenkan crosses above Kijun AND price above cloud AND price > 1d EMA50 (uptrend)
            long_entry = tenkan_kijun_cross and (curr_close > cloud_top) and (curr_close > ema_trend)
            # Short: Tenkan crosses below Kijun AND price below cloud AND price < 1d EMA50 (downtrend)
            short_entry = tenkan_kijun_cross_down and (curr_close < cloud_bottom) and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price re-enters cloud (below cloud top) OR Tenkan crosses below Kijun
            if (curr_close < cloud_top) or tenkan_kijun_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price re-enters cloud (above cloud bottom) OR Tenkan crosses above Kijun
            if (curr_close > cloud_bottom) or tenkan_kijun_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0