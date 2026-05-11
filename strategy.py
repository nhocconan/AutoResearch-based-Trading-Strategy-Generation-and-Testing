#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Breakout_With_1wTrend_Filter
# Hypothesis: Ichimoku Cloud (tenkan-sen, kijun-sen, senkou span A/B) on 6h provides dynamic support/resistance.
# Long when price breaks above cloud with bullish TK cross, aligned with weekly trend (price > weekly EMA50).
# Short when price breaks below cloud with bearish TK cross, aligned with weekly trend (price < weekly EMA50).
# Uses volume confirmation to avoid false breakouts. Designed for low turnover in ranging/trending markets.

name = "6h_Ichimoku_Cloud_Breakout_With_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # === Ichimoku Components (6h) ===
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max()
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # === Weekly Trend Filter (1w EMA50) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Ichimoku)
    start_idx = 52  # covers Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(ema50_1w_6h[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = min(senkou_span_a[i], senkou_span_b[i])
        
        # TK Cross
        tk_bullish = tenkan_sen[i] > kijun_sen[i]
        tk_bearish = tenkan_sen[i] < kijun_sen[i]
        
        if position == 0:
            # Long: Price above cloud + bullish TK cross + above weekly EMA50 + volume spike
            if (close[i] > cloud_top and tk_bullish and 
                close[i] > ema50_1w_6h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Price below cloud + bearish TK cross + below weekly EMA50 + volume spike
            elif (close[i] < cloud_bottom and tk_bearish and 
                  close[i] < ema50_1w_6h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions: TK cross reversal or price re-enters cloud
            if position == 1:
                # Exit: Bearish TK cross or price re-enters cloud
                if not tk_bullish or close[i] < cloud_top:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Bullish TK cross or price re-enters cloud
                if not tk_bearish or close[i] > cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals