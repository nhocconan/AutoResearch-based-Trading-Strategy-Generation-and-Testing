#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1d
Hypothesis: 6-hour Ichimoku Tenkan-Kijun cross with daily cloud filter (price above/below cloud) and volume confirmation (1.5x average). Uses discrete position sizing (0.25) for risk management. Ichimoku provides trend, momentum, and support/resistance in one system. Daily cloud filter ensures alignment with higher timeframe trend, reducing false signals. Volume confirmation ensures breakouts have participation. Designed for low-to-moderate trade frequency (target 12-30/year) to minimize fee drag while capturing medium-term swings in both bull and bear markets.
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
    
    # Get daily data for HTF cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Ichimoku calculations
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Calculate daily Ichimoku cloud for HTF filter
    # Daily Tenkan-sen (9-period)
    dh_high_9 = pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max().values
    dh_low_9 = pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min().values
    d_tenkan = (dh_high_9 + dh_low_9) / 2
    
    # Daily Kijun-sen (26-period)
    dh_high_26 = pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max().values
    dh_low_26 = pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min().values
    d_kijun = (dh_high_26 + dh_low_26) / 2
    
    # Daily Senkou Span A
    d_senkou_a = ((d_tenkan + d_kijun) / 2)
    
    # Daily Senkou Span B (52-period)
    dh_high_52 = pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max().values
    dh_low_52 = pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min().values
    d_senkou_b = ((dh_high_52 + dh_low_52) / 2)
    
    # Align daily cloud components to 6h (need to shift for forward-looking nature)
    # Senkou Span A and B are plotted 26 periods ahead, so to get current cloud we use values shifted back 26
    # But for filtering, we want to know if price is above/below the current cloud
    # The current cloud at time t is formed by Senkou Span A and B from 26 periods ago
    d_senkou_a_lagged = np.roll(d_senkou_a, 26)
    d_senkou_b_lagged = np.roll(d_senkou_b, 26)
    # Set first 26 values to NaN since we don't have the data yet
    d_senkou_a_lagged[:26] = np.nan
    d_senkou_b_lagged[:26] = np.nan
    
    # Align to 6h timeframe
    d_senkou_a_aligned = align_htf_to_ltf(prices, df_1d, d_senkou_a_lagged)
    d_senkou_b_aligned = align_htf_to_ltf(prices, df_1d, d_senkou_b_lagged)
    
    # Calculate volume spike filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52 for Senkou B) + volume MA
    start_idx = max(52, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(d_senkou_a_aligned[i]) or np.isnan(d_senkou_b_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine cloud boundaries (upper and lower band)
        cloud_top = max(d_senkou_a_aligned[i], d_senkou_b_aligned[i])
        cloud_bottom = min(d_senkou_a_aligned[i], d_senkou_b_aligned[i])
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: TK cross up AND price above cloud AND volume spike
            long_signal = tk_cross_up and price_above_cloud and vol_spike
            
            # Short: TK cross down AND price below cloud AND volume spike
            short_signal = tk_cross_down and price_below_cloud and vol_spike
            
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
            # Exit: TK cross down OR price falls below cloud
            if tk_cross_down or not price_above_cloud:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross up OR price rises above cloud
            if tk_cross_up or not price_below_cloud:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1d"
timeframe = "6h"
leverage = 1.0