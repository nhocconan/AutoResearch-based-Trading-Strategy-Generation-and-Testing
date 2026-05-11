#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Breakout
# Hypothesis: Uses daily Ichimoku cloud (Tenkan/Kijun + Senkou Span A/B) as trend filter and
# 6h price breaks above/below cloud with momentum confirmation (TK cross) for entries.
# Works in bull markets by buying cloud breakouts in uptrends, and in bear markets by
# selling breakdowns in downtrends. The cloud acts as dynamic support/resistance.
# Daily timeframe ensures fewer, higher-quality signals to avoid fee drag.

name = "6h_Ichimoku_Cloud_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Ichimoku components ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() +
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() +
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    senkou_span_b = (pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() +
                     pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    cloud_bottom = np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    
    # TK Cross (6h) for momentum confirmation
    tk_cross = (tenkan_sen_6h - kijun_sen_6h)
    tk_cross_prev = np.roll(tk_cross, 1)
    tk_cross_prev[0] = 0
    tk_cross_up = (tk_cross > 0) & (tk_cross_prev <= 0)
    tk_cross_down = (tk_cross < 0) & (tk_cross_prev >= 0)
    
    # Volume confirmation: volume > 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Ichimoku (52) and TK cross
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tk_cross[i]) or np.isnan(tk_cross_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 0:
            # Long: price breaks above cloud + TK cross up + volume surge
            if price_above_cloud and tk_cross_up[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud + TK cross down + volume surge
            elif price_below_cloud and tk_cross_down[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below cloud OR TK cross down
                if close[i] < cloud_top[i] or tk_cross_down[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above cloud OR TK cross up
                if close[i] > cloud_bottom[i] or tk_cross_up[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals