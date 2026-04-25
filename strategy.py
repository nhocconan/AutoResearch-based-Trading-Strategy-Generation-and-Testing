#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend
Hypothesis: Ichimoku TK cross with 1d trend filter and cloud confirmation on 6h timeframe.
Long when Tenkan > Kijun, price above cloud, and 1d EMA50 uptrend.
Short when Tenkan < Kijun, price below cloud, and 1d EMA50 downtrend.
Uses discrete position sizing (0.25) to minimize fee drag. Targets 12-30 trades/year.
Works in bull via trend following and bear via short signals aligned with 1d trend.
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
    
    # 1d data for trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter (loaded ONCE)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku calculations on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou + min_low_senkou) / 2
    
    # Cloud (Kumo): Senkou Span A and B shifted forward 26 periods
    # For signal at time i, we use Senkou values from i-26 (already published)
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Ichimoku (52) and 1d EMA (50)
    start_idx = max(52, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a_lagged[i]) or np.isnan(senkou_b_lagged[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Cloud boundaries (Senkou A and B)
        upper_cloud = max(senkou_a_lagged[i], senkou_b_lagged[i])
        lower_cloud = min(senkou_a_lagged[i], senkou_b_lagged[i])
        
        if position == 0:
            # Look for entry signals
            tk_bullish = tenkan[i] > kijun[i]
            tk_bearish = tenkan[i] < kijun[i]
            
            price_above_cloud = curr_close > upper_cloud
            price_below_cloud = curr_close < lower_cloud
            
            # Trend filter: price must be on correct side of 1d EMA50
            long_trend = curr_close > ema_50_1d_aligned[i]
            short_trend = curr_close < ema_50_1d_aligned[i]
            
            long_entry = tk_bullish and price_above_cloud and long_trend
            short_entry = tk_bearish and price_below_cloud and short_trend
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when TK cross turns bearish OR price closes below cloud
            if tenkan[i] < kijun[i] or curr_close < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when TK cross turns bullish OR price closes above cloud
            if tenkan[i] > kijun[i] or curr_close > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0