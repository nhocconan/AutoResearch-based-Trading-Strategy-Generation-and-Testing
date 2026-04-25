#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: Ichimoku Tenkan/Kijun cross with cloud filter on 6h, aligned with 1d trend (price >/< Kumo twist) captures medium-term momentum with low whipsaw. 
Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (26/52-period). 
Long when Tenkan > Kijun, price above cloud, and 1d bullish (price > Senkou Span A 1d ago). 
Short when Tenkan < Kijun, price below cloud, and 1d bearish (price < Senkou Span B 1d ago). 
Uses discrete sizing (0.25) to limit ~30-60 trades/year and minimize fee drag. 
Works in bull/bear by requiring cloud alignment and 1d trend filter, avoiding counter-trend whipsaws.
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
    
    # 1d data for Ichimoku and trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align 1d Ichimoku to 6h (wait for completed 1d bar + Senkou Span shift)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # 1d trend filter: price vs Senkou Span from prior day (to avoid look-ahead)
    # Use Senkou Span A/B from 1d ago for trend filter
    senkou_a_prev = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=1)
    senkou_b_prev = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=1)
    
    # 6h ATR for stop loss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need Ichimoku warmup (52) + ATR (14)
    start_idx = max(52, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(senkou_a_prev[i]) or np.isnan(senkou_b_prev[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        lower_cloud = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        # 1d trend filter: price vs prior day's Senkou Span
        bullish_1d = curr_close > senkou_a_prev[i]
        bearish_1d = curr_close < senkou_b_prev[i]
        
        if position == 0:
            # Long: Tenkan > Kijun, price above cloud, 1d bullish
            long_signal = (tenkan_6h[i] > kijun_6h[i]) and (curr_close > upper_cloud) and bullish_1d
            # Short: Tenkan < Kijun, price below cloud, 1d bearish
            short_signal = (tenkan_6h[i] < kijun_6h[i]) and (curr_close < lower_cloud) and bearish_1d
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Tenkan < Kijun OR price below cloud OR ATR stoploss hit
            if (tenkan_6h[i] < kijun_6h[i]) or (curr_close < lower_cloud) or (curr_close < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Tenkan > Kijun OR price above cloud OR ATR stoploss hit
            if (tenkan_6h[i] > kijun_6h[i]) or (curr_close > upper_cloud) or (curr_close > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0