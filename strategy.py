#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_Direction_v1
Hypothesis: Ichimoku cloud breakout with weekly trend filter on 6h timeframe. 
In bull markets: price above cloud + TK cross up + weekly trend up = long.
In bear markets: price below cloud + TK cross down + weekly trend down = short.
Volume confirmation reduces false signals. Designed for low trade frequency (~15-30/year) 
to minimize fee drag while capturing strong trends in both regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 52 or len(df_1d) < 52:
        return np.zeros(n)
    
    # === Ichimoku Cloud components (9, 26, 52 periods) on 1d data ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # === Weekly trend filter: 34-period EMA on 1w ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Volume confirmation (20-period on 1d) ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        weekly_trend = ema_34_1w_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        
        # Cloud boundaries (Senkou Span A and B form the cloud)
        upper_cloud = max(span_a, span_b)
        lower_cloud = min(span_a, span_b)
        
        if position == 0:
            # Long: price above cloud + TK cross up (Tenkan > Kijun) + weekly trend up + volume confirmation
            if (price_close > upper_cloud and tenkan > kijun and 
                weekly_trend > price_close and vol_spike > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross down (Tenkan < Kijun) + weekly trend down + volume confirmation
            elif (price_close < lower_cloud and tenkan < kijun and 
                  weekly_trend < price_close and vol_spike > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: TK cross in opposite direction OR price re-enters cloud
            if position == 1:  # Long position
                if tenkan < kijun or price_close < upper_cloud:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Short position
                if tenkan > kijun or price_close > lower_cloud:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_Direction_v1"
timeframe = "6h"
leverage = 1.0