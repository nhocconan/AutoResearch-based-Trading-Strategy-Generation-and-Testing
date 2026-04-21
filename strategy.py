#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_HTFConfirm_v1
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) on 6h with 1-week EMA50 trend filter and volume spike confirmation. Cloud twist indicates momentum shift; 1w EMA50 filters for major trend alignment; volume spike confirms institutional participation. Designed for low trade frequency (~15-25/year) to work in both bull/bear markets by requiring strong trend alignment and momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1w trend filter: 50-period EMA ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === Ichimoku components on 6h (9, 26, 52 periods) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Shift Senkou Spans forward by 26 periods
    senkou_span_a_shifted = np.roll(senkou_span_a, 26)
    senkou_span_b_shifted = np.roll(senkou_span_b, 26)
    senkou_span_a_shifted[:26] = np.nan
    senkou_span_b_shifted[:26] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a_shifted[i]) or np.isnan(senkou_span_b_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1w = ema_50_1w_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        span_a = senkou_span_a_shifted[i]
        span_b = senkou_span_b_shifted[i]
        
        # Cloud twist detection: Senkou Span A/B cross
        # Bullish twist: Span A crosses above Span B (previous Span A <= Span B, current Span A > Span B)
        # Bearish twist: Span A crosses below Span B (previous Span A >= Span B, current Span A < Span B)
        if i > 0:
            prev_span_a = senkou_span_a_shifted[i-1]
            prev_span_b = senkou_span_b_shifted[i-1]
            bullish_twist = (prev_span_a <= prev_span_b) and (span_a > span_b)
            bearish_twist = (prev_span_a >= prev_span_b) and (span_a < span_b)
        else:
            bullish_twist = False
            bearish_twist = False
        
        if position == 0:
            # Long: bullish cloud twist + price above 1w EMA50 + volume spike > 2.0
            if bullish_twist and price_close > trend_1w and vol_spike > 2.0:
                signals[i] = 0.25
                position = 1
            # Short: bearish cloud twist + price below 1w EMA50 + volume spike > 2.0
            elif bearish_twist and price_close < trend_1w and vol_spike > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit on opposite cloud twist or price crosses Kijun-sen
            if position == 1:
                # Exit long: bearish twist or price closes below Kijun-sen
                if bearish_twist or price_close < kijun:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: bullish twist or price closes above Kijun-sen
                if bullish_twist or price_close > kijun:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1wTrend_HTFConfirm_v1"
timeframe = "6h"
leverage = 1.0