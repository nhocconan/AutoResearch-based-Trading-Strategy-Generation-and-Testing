#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_ADXFilter_v1
Hypothesis: Ichimoku TK cross (tenkan/kijun) with Kumo twist (senkou A/B cross) on 6h, filtered by 1w ADX trend strength (>25) and 1d price above/below Kumo. Only trades in strong trends with momentum confirmation. Designed for low turnover (~15-25 trades/year) to avoid fee drag. Works in bull/bear by only trading with 1w trend direction confirmed by ADX and Kumo twist.
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
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend (ADX)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w ADX(14) for trend strength filter
    # Calculate ADX components
    plus_dm = np.zeros(len(df_1w))
    minus_dm = np.zeros(len(df_1w))
    tr = np.zeros(len(df_1w))
    
    for i in range(1, len(df_1w)):
        high_diff = df_1w['high'].iloc[i] - df_1w['high'].iloc[i-1]
        low_diff = df_1w['low'].iloc[i-1] - df_1w['low'].iloc[i]
        
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(
            df_1w['high'].iloc[i] - df_1w['low'].iloc[i],
            abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1]),
            abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1])
        )
    
    tr[0] = df_1w['high'].iloc[0] - df_1w['low'].iloc[0]
    
    # Smooth with Wilder's smoothing (EMA with alpha=1/period)
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di_1w = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w + 1e-10)
    adx_1w = pd.Series(dx_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Get 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:  # need 26*2 for senkou
        return np.zeros(n)
    
    # Ichimoku calculations on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current Kumo (cloud) edges: Senkou A/B shifted back 26 periods to align with price
    # We need the cloud that was plotted 26 periods ago (i.e., current cloud)
    senkou_a_lag = np.roll(senkou_a, 26)
    senkou_b_lag = np.roll(senkou_b, 26)
    senkou_a_lag[:26] = np.nan
    senkou_b_lag[:26] = np.nan
    
    # Kumo twist: when Senkou A crosses Senkou B (trend change signal)
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    senkou_a_prev = np.roll(senkou_a_lag, 1)
    senkou_b_prev = np.roll(senkou_b_lag, 1)
    senkou_a_prev[0] = np.nan
    senkou_b_prev[0] = np.nan
    
    bullish_twist = (senkou_a_lag > senkou_b_lag) & (senkou_a_prev <= senkou_b_prev)
    bearish_twist = (senkou_a_lag < senkou_b_lag) & (senkou_a_prev >= senkou_b_prev)
    
    # Align HTF indicators to 6h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_lag)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_lag)
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of ADX(14) needs ~30, Ichimoku needs 52+26=78
    start_idx = max(30, 78) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1w_aligned[i]) or 
            np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(bullish_twist_aligned[i]) or
            np.isnan(bearish_twist_aligned[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_1w_aligned[i]
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        bullish_twist_val = bullish_twist_aligned[i]
        bearish_twist_val = bearish_twist_aligned[i]
        
        # Trend filter: only trade when 1w ADX > 25 (strong trend)
        strong_trend = adx_val > 25
        
        # Price relative to Kumo: above cloud = bullish bias, below cloud = bearish bias
        kumo_top = max(senkou_a_val, senkou_b_val)
        kumo_bottom = min(senkou_a_val, senkou_b_val)
        price_above_kumo = close_val > kumo_top
        price_below_kumo = close_val < kumo_bottom
        
        # TK cross: Tenkan crosses Kijun
        tenkan_prev = np.roll(tenkan_aligned, 1)[i]
        kijun_prev = np.roll(kijun_aligned, 1)[i]
        if i == 0:
            tenkan_prev = tenkan_val
            kijun_prev = kijun_val
        
        tk_cross_up = (tenkan_val > kijun_val) and (tenkan_prev <= kijun_prev)
        tk_cross_down = (tenkan_val < kijun_val) and (tenkan_prev >= kijun_prev)
        
        if position == 0:
            # Long: bullish TK cross + price above Kumo + strong trend + bullish Kumo twist (confirmation)
            long_signal = tk_cross_up and \
                         price_above_kumo and \
                         strong_trend and \
                         (bullish_twist_val > 0.5)
            
            # Short: bearish TK cross + price below Kumo + strong trend + bearish Kumo twist (confirmation)
            short_signal = tk_cross_down and \
                          price_below_kumo and \
                          strong_trend and \
                          (bearish_twist_val > 0.5)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long: exit on bearish TK cross or price drops below Kumo
            signals[i] = 0.25
            exit_signal = tk_cross_down or (close_val < kumo_bottom)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short: exit on bullish TK cross or price rises above Kumo
            signals[i] = -0.25
            exit_signal = tk_cross_up or (close_val > kumo_top)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1wTrend_ADXFilter_v1"
timeframe = "6h"
leverage = 1.0