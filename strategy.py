#!/usr/bin/env python3
# 6h_ichimoku_cloud_1w_trend_volume_v1
# Hypothesis: 6h Ichimoku cloud breakout with 1w trend filter (EMA50) and volume confirmation.
# Works in bull/bear: 1w EMA50 defines institutional trend; Ichimoku cloud provides dynamic support/resistance;
# TK cross confirms momentum; volume confirms institutional participation. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA trend and Ichimoku calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Ichimoku components (based on completed 1w candles)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe (completed 1w bar only)
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR trend turns bearish
            if close[i] < cloud_bottom or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR trend turns bullish
            if close[i] > cloud_top or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.8 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above cloud with bullish trend and TK cross up
                if (close[i] > cloud_top and 
                    close[i] > ema50_1w_aligned[i] and
                    tenkan_aligned[i] > kijun_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below cloud with bearish trend and TK cross down
                elif (close[i] < cloud_bottom and 
                      close[i] < ema50_1w_aligned[i] and
                      tenkan_aligned[i] < kijun_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals