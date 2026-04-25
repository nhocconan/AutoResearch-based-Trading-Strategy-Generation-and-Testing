#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_Filter
Hypothesis: 6h Ichimoku TK cross with Kumo twist (Senkou A/B crossover) filtered by 1w trend (price > 200 EMA).
Goes long when TK cross bullish + price above Kumo + 1w uptrend, short when TK cross bearish + price below Kumo + 1w downtrend.
Uses discrete sizing (0.25) to minimize fees. Target: 12-37 trades/year on 6h.
Works in bull via trend continuation, in bear via counter-trend pulls to Kumo with 1w trend filter avoiding false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1w data for trend filter (200 EMA on weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (no extra delay needed for these)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # Kumo twist: Senkou A crossing above/below Senkou B
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    senkou_a_prev = np.concatenate([[senkou_a_aligned[0]], senkou_a_aligned[:-1]])
    senkou_b_prev = np.concatenate([[senkou_b_aligned[0]], senkou_b_aligned[:-1]])
    
    bullish_twist = (senkou_a_aligned > senkou_b_aligned) & (senkou_a_prev <= senkou_b_prev)
    bearish_twist = (senkou_a_aligned < senkou_b_aligned) & (senkou_a_prev >= senkou_b_prev)
    
    # TK cross: Tenkan crossing above/below Kijun
    tenkan_prev = np.concatenate([[tenkan_aligned[0]], tenkan_aligned[:-1]])
    kijun_prev = np.concatenate([[kijun_aligned[0]], kijun_aligned[:-1]])
    
    tk_bullish_cross = (tenkan_aligned > kijun_aligned) & (tenkan_prev <= kijun_prev)
    tk_bearish_cross = (tenkan_aligned < kijun_aligned) & (tenkan_prev >= kijun_prev)
    
    # Price relative to Kumo (cloud)
    # Price above Kumo: price > max(Senkou A, Senkou B)
    # Price below Kumo: price < min(Senkou A, Senkou B)
    kumou_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    kumou_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    price_above_kumo = close > kumou_top
    price_below_kumo = close < kumou_bottom
    
    # 1w trend filter: price > 200 EMA for uptrend, price < 200 EMA for downtrend
    trend_up = close > ema_200_1w_aligned
    trend_down = close < ema_200_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku calculations (max 52 periods)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: TK bullish cross + price above Kumo + 1w uptrend
            long_signal = tk_bullish_cross[i] and price_above_kumo[i] and trend_up[i]
            # Short: TK bearish cross + price below Kumo + 1w downtrend
            short_signal = tk_bearish_cross[i] and price_below_kumo[i] and trend_down[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when TK bearish cross OR price closes below Kumo (cloud break)
            exit_signal = tk_bearish_cross[i] or (close[i] < kumou_bottom[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when TK bullish cross OR price closes above Kumo (cloud break)
            exit_signal = tk_bullish_cross[i] or (close[i] > kumou_top[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0