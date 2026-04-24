#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
- Primary timeframe: 6h for Ichimoku calculation and entries.
- HTF: 1d for trend direction (price above/below Kumo cloud from 1d).
- Volume: Current 6h volume > 1.5 * 20-period volume MA to confirm breakouts.
- Entry: Long when Tenkan-sen crosses above Kijun-sen AND price is above Kumo (bullish) AND volume spike.
         Short when Tenkan-sen crosses below Kijun-sen AND price is below Kumo (bearish) AND volume spike.
- Exit: Opposite Tenkan/Kijun cross or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Why it works: Ichimoku provides dynamic support/resistance (Kumo), trend (Tenkan/Kijun cross), and momentum.
              Works in bull markets via breakouts above cloud, in bear markets via breakdowns below cloud.
              Volume filter reduces false signals in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components on 6h
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
    
    # Get 1d data for Kumo (cloud) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components for cloud
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (period26_high_1d + period26_low_1d) / 2
    
    senkou_span_a_1d = (tenkan_sen_1d + kijun_sen_1d) / 2
    
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Kumo cloud boundaries: max/min of Senkou Span A and B
    kumo_top_1d = np.maximum(senkou_span_a_1d, senkou_span_b_1d)
    kumo_bottom_1d = np.minimum(senkou_span_a_1d, senkou_span_b_1d)
    
    # Align 1d cloud to 6h
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period volume MA
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 26, 20)  # Need enough bars for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Tenkan/Kijun cross signals
        tenkan_cross_above = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tenkan_cross_below = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        price = close[i]
        kumo_top = kumo_top_aligned[i]
        kumo_bottom = kumo_bottom_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Tenkan crosses above Kijun AND price above Kumo (bullish cloud)
                if tenkan_cross_above and price > kumo_top:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Tenkan crosses below Kijun AND price below Kumo (bearish cloud)
                elif tenkan_cross_below and price < kumo_bottom:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR loss of volume confirmation
            if tenkan_cross_below or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR loss of volume confirmation
            if tenkan_cross_above or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1dKumoTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0