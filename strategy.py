#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm
Hypothesis: 6h Ichimoku Kumo twist (Senkou Span A/B cross) with 1d trend filter (price >/<- 200 EMA) and volume confirmation.
Long when Senkou Span A crosses above Senkou Span B (bullish Kumo twist) AND price > 1d EMA200 AND volume > 1.5x 20-period average.
Short when Senkou Span A crosses below Senkou Span B (bearish Kumo twist) AND price < 1d EMA200 AND volume > 1.5x 20-period average.
Exit on opposite Kumo twist or loss of 1d EMA200 alignment.
Ichimoku cloud twist captures momentum shifts; 1d EMA200 filters for higher-timeframe trend alignment; volume confirms conviction.
Works in bull markets (bullish twists with uptrend) and bear markets (bearish twists with downtrend).
Target: 50-150 total trades over 4 years = 12-37/year.
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
    
    # Calculate Ichimoku components from 6h data
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
    
    # Kumo twist signals: Senkou Span A crossing Senkou Span B
    # Bullish twist: Senkou Span A crosses above Senkou Span B
    bullish_twist = (senkou_span_a > senkou_span_b) & (np.roll(senkou_span_a, 1) <= np.roll(senkou_span_b, 1))
    # Bearish twist: Senkou Span A crosses below Senkou Span B
    bearish_twist = (senkou_span_a < senkou_span_b) & (np.roll(senkou_span_a, 1) >= np.roll(senkou_span_b, 1))
    
    # 1d EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need enough for Ichimoku (52 periods) and 1d EMA200
    start_idx = max(52, 200)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_200_aligned[i]
        vol_conf = volume_confirm[i]
        bullish = bullish_twist[i]
        bearish = bearish_twist[i]
        
        if position == 0:
            # Flat - look for entry: Kumo twist with 1d EMA200 alignment and volume confirmation
            # Long: Bullish Kumo twist AND price > 1d EMA200 AND volume confirmation
            # Short: Bearish Kumo twist AND price < 1d EMA200 AND volume confirmation
            long_condition = bullish and (close_val > ema_val) and vol_conf
            short_condition = bearish and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long - exit when bearish Kumo twist OR loses 1d EMA200 alignment
            if bearish or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when bullish Kumo twist OR loses 1d EMA200 alignment
            if bullish or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0