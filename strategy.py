#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_WeeklyRegime
Hypothesis: Ichimoku TK cross with Kumo twist confirmation on 6h, filtered by 1d trend (price vs Kumo) and weekly regime (ADX < 20 = range, ADX > 25 = trend). 
In trend regime: trade TK cross in direction of 1d Kumo (price above/below Kumo). 
In range regime: fade TK cross at Kumo edges (price at Kumo top/bottom). 
Designed for 12-25 trades/year (50-100 over 4 years) to minimize fee drag.
Uses discrete position sizing (0.25) and ATR-based stoploss.
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
    
    # Ichimoku parameters (9, 26, 52)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = (tenkan + kijun) / 2
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # The Kumo (cloud) is between Senkou Span A and Senkou Span B
    # We need current cloud values (displaced forward by 26 periods)
    # So we use values calculated 26 periods ago for current cloud
    if len(senkou_span_a) > displacement:
        senkou_span_a_current = np.roll(senkou_span_a, displacement)
        senkou_span_b_current = np.roll(senkou_span_b, displacement)
        # First 'displacement' values are invalid
        senkou_span_a_current[:displacement] = np.nan
        senkou_span_b_current[:displacement] = np.nan
    else:
        senkou_span_a_current = np.full_like(senkou_span_a, np.nan)
        senkou_span_b_current = np.full_like(senkou_span_b, np.nan)
    
    # Kumo top (max of A and B) and Kumo bottom (min of A and B)
    kumo_top = np.maximum(senkou_span_a_current, senkou_span_b_current)
    kumo_bottom = np.minimum(senkou_span_a_current, senkou_span_b_current)
    
    # TK Cross: Tenkan crossing above/below Kijun
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Price relative to Kumo
    price_above_kumo = close > kumo_top
    price_below_kumo = close < kumo_bottom
    price_in_kumo = (close >= kumo_bottom) & (close <= kumo_top)
    
    # 1d data for trend filter (price vs 1d Kumo)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku for trend filter
    highest_1d_tenkan = pd.Series(df_1d['high']).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_1d_tenkan = pd.Series(df_1d['low']).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_1d = (highest_1d_tenkan + lowest_1d_tenkan) / 2
    
    highest_1d_kijun = pd.Series(df_1d['high']).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_1d_kijun = pd.Series(df_1d['low']).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_1d = (highest_1d_kijun + lowest_1d_kijun) / 2
    
    senkou_span_a_1d = (tenkan_1d + kijun_1d) / 2
    
    highest_1d_senkou_b = pd.Series(df_1d['high']).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_1d_senkou_b = pd.Series(df_1d['low']).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b_1d = (highest_1d_senkou_b + lowest_1d_senkou_b) / 2
    
    # Current 1d Kumo (displaced)
    if len(senkou_span_a_1d) > displacement:
        senkou_span_a_1d_current = np.roll(senkou_span_a_1d, displacement)
        senkou_span_b_1d_current = np.roll(senkou_span_b_1d, displacement)
        senkou_span_a_1d_current[:displacement] = np.nan
        senkou_span_b_1d_current[:displacement] = np.nan
    else:
        senkou_span_a_1d_current = np.full_like(senkou_span_a_1d, np.nan)
        senkou_span_b_1d_current = np.full_like(senkou_span_b_1d, np.nan)
    
    kumo_top_1d = np.maximum(senkou_span_a_1d_current, senkou_span_b_1d_current)
    kumo_bottom_1d = np.minimum(senkou_span_a_1d_current, senkou_span_b_1d_current)
    
    # 1d trend: price above/below 1d Kumo
    price_above_1d_kumo = close > kumo_top_1d
    price_below_1d_kumo = close < kumo_bottom_1d
    
    # Align 1d Kumo to 6h timeframe
    kumo_top_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    price_above_1d_kumo_aligned = align_htf_to_ltf(prices, df_1d, price_above_1d_kumo.astype(float))
    price_below_1d_kumo_aligned = align_htf_to_ltf(prices, df_1d, price_below_1d_kumo.astype(float))
    
    # Weekly regime filter (ADX < 20 = range, ADX > 25 = trend)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX on weekly data
    period = 14
    plus_dm = np.diff(df_1w['high'].values, prepend=df_1w['high'].values[0])
    minus_dm = np.diff(df_1w['low'].values, prepend=df_1w['low'].values[0])
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = np.abs(np.diff(df_1w['high'].values, prepend=df_1w['high'].values[0]))
    tr2 = np.abs(np.diff(df_1w['low'].values, prepend=df_1w['low'].values[0]))
    tr0 = np.abs(df_1w['high'].values - df_1w['low'].values)
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    
    atr_period = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values / atr_period)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values / atr_period)
    dx = (np.abs(plus_di - minus_di) / (np.maximum(plus_di, minus_di) + 1e-10)) * 100
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Weekly regime: ADX < 20 = range, ADX > 25 = trend
    weekly_range = adx_aligned < 20
    weekly_trend = adx_aligned > 25
    
    # ATR for stoploss calculation (6h)
    tr0_6h = np.abs(high - low)
    tr1_6h = np.abs(np.diff(high, prepend=high[0]))
    tr2_6h = np.abs(np.diff(low, prepend=low[0]))
    tr_6h = np.maximum(tr0_6h, np.maximum(tr1_6h, tr2_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Ichimoku (52+26), 1d Ichimoku, weekly ADX
    start_idx = max(senkou_span_b_period + displacement, 100)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(kumo_top[i]) or 
            np.isnan(kumo_bottom[i]) or np.isnan(kumo_top_1d_aligned[i]) or 
            np.isnan(kumo_bottom_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Look for entry signals
            long_tk_cross = tk_cross_up[i]
            short_tk_cross = tk_cross_down[i]
            
            if weekly_range[i]:
                # Range regime: fade TK cross at Kumo edges
                long_entry = long_tk_cross and price_below_kumo[i] and curr_low <= kumo_bottom[i] * 1.001
                short_entry = short_tk_cross and price_above_kumo[i] and curr_high >= kumo_top[i] * 0.999
            else:
                # Trend regime: trade TK cross in direction of 1d Kumo
                long_entry = long_tk_cross and price_above_1d_kumo_aligned[i]
                short_entry = short_tk_cross and price_below_1d_kumo_aligned[i]
            
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
            # Long position: exit when TK cross down or price closes below Kumo bottom or ATR stoploss hit
            atr_stop = entry_price - 2.0 * atr_6h[i]
            if tk_cross_down[i] or curr_close < kumo_bottom[i] or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when TK cross up or price closes above Kumo top or ATR stoploss hit
            atr_stop = entry_price + 2.0 * atr_6h[i]
            if tk_cross_up[i] or curr_close > kumo_top[i] or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_WeeklyRegime"
timeframe = "6h"
leverage = 1.0