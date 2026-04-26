#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrendFilter_v1
Hypothesis: 6h Ichimoku strategy with daily trend filter and Kumo twist confirmation.
- Uses 6h timeframe for lower trade frequency (target: 50-150 total trades over 4 years)
- Ichimoku cloud (Senkou Span A/B) from 6h data for trend and support/resistance
- TK line (Tenkan/Kijun) cross for entry timing
- Kumo twist (Senkou Span A/B cross) as trend change filter from 1d data
- Long when: price > cloud, TK cross up, and 1d Kumo bullish (Senkou A > Senkou B)
- Short when: price < cloud, TK cross down, and 1d Kumo bearish (Senkou A < Senkou B)
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by aligning with higher timeframe trend via Kumo twist
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for Ichimoku calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for Kumo twist filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not used for signals)
    
    # Calculate 1d Ichimoku for Kumo twist (trend filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    max_high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    min_low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (max_high_9_1d + min_low_9_1d) / 2
    
    # 1d Kijun-sen (26-period)
    max_high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    min_low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (max_high_26_1d + min_low_26_1d) / 2
    
    # 1d Senkou Span A
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # 1d Senkou Span B (52-period)
    max_high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    min_low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((max_high_52_1d + min_low_52_1d) / 2)
    
    # Kumo twist: Senkou A > Senkou B (bullish) or Senkou A < Senkou B (bearish)
    kumo_twist_bullish = senkou_a_1d > senkou_b_1d
    kumo_twist_bearish = senkou_a_1d < senkou_b_1d
    
    # Align 6h Ichimoku components (no alignment needed as calculated on same timeframe)
    # Align 1d Kumo twist to 6h timeframe (wait for completed 1d bar)
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bullish.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B calculation)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(kumo_twist_bullish_aligned[i]) or np.isnan(kumo_twist_bearish_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku conditions
        price_above_cloud = close[i] > max(senkou_a[i], senkou_b[i])
        price_below_cloud = close[i] < min(senkou_a[i], senkou_b[i])
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # 1d Kumo twist filter
        kumo_bullish = kumo_twist_bullish_aligned[i] > 0.5
        kumo_bearish = kumo_twist_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Long: price above cloud, TK cross up, and 1d Kumo bullish
            if price_above_cloud and tk_cross_up and kumo_bullish:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, TK cross down, and 1d Kumo bearish
            elif price_below_cloud and tk_cross_down and kumo_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below cloud OR TK cross down OR 1d Kumo turns bearish
            if price_below_cloud or tk_cross_down or not kumo_bullish:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above cloud OR TK cross up OR 1d Kumo turns bullish
            if price_above_cloud or tk_cross_up or not kumo_bearish:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0