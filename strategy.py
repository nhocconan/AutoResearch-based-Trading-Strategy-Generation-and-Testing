#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrendFilter_v3
Hypothesis: 6h Ichimoku cloud twist (TK cross) with 1d trend filter and volume confirmation.
- Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Ichimoku: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (26/52-period)
- Long when TK crosses above AND price > cloud AND 1d uptrend AND volume spike
- Short when TK crosses below AND price < cloud AND 1d downtrend AND volume spike
- Cloud acts as dynamic support/resistance; TK cross indicates momentum shift
- Volume confirmation reduces false signals; 1d trend filter ensures alignment with higher timeframe
- Designed to capture trending moves while avoiding choppy markets via cloud filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for Ichimoku calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period_senkou_b = 52
    highest_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to current timeframe (no shift needed as they are already calculated)
    # Note: Senkou spans are naturally aligned as they are plotted ahead but we use current values
    
    # Calculate volume spike (20-period volume average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for Kijun, 9 for Tenkan, 20 for volume MA)
    start_idx = max(52, 26, 9, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema34_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku conditions
        # Determine cloud top and bottom (Senkou Span A and B)
        cloud_top = max(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = min(senkou_span_a[i], senkou_span_b[i])
        
        # TK cross conditions
        tk_cross_above = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_below = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # 1d trend filter
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: TK cross above AND price above cloud AND 1d uptrend AND volume spike
            if tk_cross_above and price_above_cloud and trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below AND price below cloud AND 1d downtrend AND volume spike
            elif tk_cross_below and price_below_cloud and trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross below OR price falls below cloud OR 1d trend turns down
            if tk_cross_below or close[i] < cloud_bottom or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross above OR price rises above cloud OR 1d trend turns up
            if tk_cross_above or close[i] > cloud_top or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrendFilter_v3"
timeframe = "6h"
leverage = 1.0