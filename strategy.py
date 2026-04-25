#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter_WeeklyTrend
Hypothesis: 6-hour Ichimoku Tenkan-Kijun cross with 1-day cloud filter (price above/below cloud) and 1-week trend filter (price above/below 1w EMA50).
Ichimoku TK cross provides timely momentum signals, 1d cloud acts as dynamic support/resistance filter, and 1w EMA50 ensures alignment with higher timeframe trend.
This combination reduces false signals in choppy markets while capturing strong trends. Works in both bull (long when price > cloud & > 1w EMA) and bear (short when price < cloud & < 1w EMA).
Target: 12-30 trades/year by requiring confluence of TK cross, cloud position, and weekly trend.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Ichimoku cloud (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    # Ichimoku components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52 displacement)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)  # 26-period shift for leading span
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)  # 26-period shift for leading span
    
    # 1w data for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations (52) + weekly EMA (50)
    start_idx = max(52, 50) + 26  # Add displacement for Senkou spans
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Ichimoku TK cross: Tenkan crosses above/below Kijun
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        # Cloud filter: price above/below cloud (Senkou Span A and B)
        # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Weekly trend filter: price above/below 1w EMA50
        price_above_1w_ema = curr_close > ema_50_1w_aligned[i]
        price_below_1w_ema = curr_close < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: TK cross up + price above cloud + price above weekly EMA
            if tk_cross_up and price_above_cloud and price_above_1w_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: TK cross down + price below cloud + price below weekly EMA
            elif tk_cross_down and price_below_cloud and price_below_1w_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: TK cross down OR price below cloud OR price below weekly EMA
            if tk_cross_down or not price_above_cloud or not price_above_1w_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross up OR price above cloud OR price above weekly EMA
            if tk_cross_up or not price_below_cloud or not price_below_1w_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_WeeklyTrend"
timeframe = "6h"
leverage = 1.0