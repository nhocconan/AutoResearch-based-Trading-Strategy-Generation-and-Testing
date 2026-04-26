#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_ADXFilter_v1
Hypothesis: Ichimoku TK cross + Kumo twist (Senkou Span A/B cross) on 6h with 1d ADX>25 trend filter and volume confirmation. Kumo twist indicates momentum shift; TK cross provides entry timing. ADX filter ensures trending markets only. Designed for BTC/ETH with ~25-40 trades/year to avoid fee drag. Works in bull/bear by only trading with strong trend (ADX>25).
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
    
    # Get 1d data for HTF trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    plus_dm = np.where((df_1d['high'].values - np.roll(df_1d['high'].values, 1)) > 
                       (np.roll(df_1d['low'].values, 1) - df_1d['low'].values),
                       np.maximum(df_1d['high'].values - np.roll(df_1d['high'].values, 1), 0), 0)
    minus_dm = np.where((np.roll(df_1d['low'].values, 1) - df_1d['low'].values) > 
                        (df_1d['high'].values - np.roll(df_1d['high'].values, 1)),
                        np.maximum(np.roll(df_1d['low'].values, 1) - df_1d['low'].values, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr = np.maximum(df_1d['high'].values - df_1d['low'].values,
                    np.maximum(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                               np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))))
    tr[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get 6h data for Ichimoku
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
    high_9 = pd.Series(df_6h['high'].values).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_6h['low'].values).rolling(window=9, min_periods=9).min().values
    high_26 = pd.Series(df_6h['high'].values).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_6h['low'].values).rolling(window=26, min_periods=26).min().values
    high_52 = pd.Series(df_6h['high'].values).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_6h['low'].values).rolling(window=52, min_periods=52).min().values
    
    tenkan_sen = (high_9 + low_9) / 2  # Conversion Line
    kijun_sen = (high_26 + low_26) / 2  # Base Line
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)  # Leading Span A
    senkou_span_b = ((high_52 + low_52) / 2)  # Leading Span B
    chikou_span = df_6h['close'].values  # Lagging Span (not used for signals)
    
    # Align HTF indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # Volume spike filter: volume > 1.8x median (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of ADX(14) 1d, Ichimoku (52), volume median (20)
    start_idx = max(30, 52, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_median[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        adx_val = adx_1d_aligned[i]
        close_val = close[i]
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        senkou_a_val = senkou_span_a_aligned[i]
        senkou_b_val = senkou_span_b_aligned[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        # Volume confirmation: only trade in above-average volume
        volume_ok = volume_val > 1.8 * vol_median_val
        
        # Kumo twist: Senkou Span A crosses above/below Senkou Span B
        # We need previous bar values to detect cross
        if i > start_idx:
            prev_senkou_a = senkou_span_a_aligned[i-1]
            prev_senkou_b = senkou_span_b_aligned[i-1]
            kumo_twist_up = (senkou_a_val > senkou_b_val) and (prev_senkou_a <= prev_senkou_b)
            kumo_twist_down = (senkou_a_val < senkou_b_val) and (prev_senkou_a >= prev_senkou_b)
        else:
            kumo_twist_up = False
            kumo_twist_down = False
        
        # TK cross: Tenkan crosses Kijun
        if i > start_idx:
            prev_tenkan = tenkan_sen_aligned[i-1]
            prev_kijun = kijun_sen_aligned[i-1]
            tk_cross_up = (tenkan_val > kijun_val) and (prev_tenkan <= prev_kijun)
            tk_cross_down = (tenkan_val < kijun_val) and (prev_tenkan >= prev_kijun)
        else:
            tk_cross_up = False
            tk_cross_down = False
        
        if position == 0:
            # Long: Kumo twist up + TK cross up + strong trend + volume
            long_signal = kumo_twist_up and tk_cross_up and strong_trend and volume_ok
            
            # Short: Kumo twist down + TK cross down + strong trend + volume
            short_signal = kumo_twist_down and tk_cross_down and strong_trend and volume_ok
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross down OR Kumo twist down OR ADX falls below 20
            if tk_cross_down or kumo_twist_down or adx_val < 20:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross up OR Kumo twist up OR ADX falls below 20
            if tk_cross_up or kumo_twist_up or adx_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_ADXFilter_v1"
timeframe = "6h"
leverage = 1.0