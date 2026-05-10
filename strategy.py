#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_1dTrend_Filter
# Hypothesis: Ichimoku system (Tenkan/Kijun cross + Cloud) identifies trend direction on 6h,
# filtered by 1d Ichimoku Cloud color (bullish/bearish) to avoid counter-trend trades.
# Works in bull markets via trend continuation and in bear markets via counter-trend bounces
# when price rejects the 1d cloud in the opposite direction.
# Entry: TK cross + price above/below 6h cloud + 1d cloud filter.
# Exit: TK cross reversal or price re-enters 6h cloud.
# Target: 50-150 total trades over 4 years.

name = "6h_Ichimoku_Cloud_1dTrend_Filter"
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
    
    # 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max()
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): close shifted -22 periods (not used in signals)
    
    # 1d Ichimoku for trend filter (Cloud color only)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan and Kijun
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    
    # 1d Senkou Span A and B
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                    pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    
    # 1d Cloud color: bullish if Senkou A > Senkou B, bearish if Senkou A < Senkou B
    cloud_bullish_1d = senkou_a_1d > senkou_b_1d
    cloud_bearish_1d = senkou_a_1d < senkou_b_1d
    
    # Align 1d cloud colors to 6h
    cloud_bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_bullish_1d.astype(float))
    cloud_bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_bearish_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are valid
    start_idx = max(52, 26) + 26  # Senkou B needs 52 periods, plus 26 shift
    
    for i in range(start_idx, n):
        # Skip if any NaN values
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(cloud_bullish_1d_aligned[i]) or np.isnan(cloud_bearish_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 6h cloud color (for price vs cloud check)
        cloud_bullish_6h = senkou_a[i] > senkou_b[i]
        cloud_bearish_6h = senkou_a[i] < senkou_b[i]
        
        # TK cross signals
        tk_cross_bull = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_bear = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        if position == 0:
            # Long: bullish TK cross + price above 6h cloud + 1d bullish cloud
            if tk_cross_bull and close[i] > max(senkou_a[i], senkou_b[i]) and cloud_bullish_1d_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross + price below 6h cloud + 1d bearish cloud
            elif tk_cross_bear and close[i] < min(senkou_a[i], senkou_b[i]) and cloud_bearish_1d_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish TK cross OR price re-enters 6h cloud
            if tk_cross_bear or close[i] <= max(senkou_a[i], senkou_b[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish TK cross OR price re-enters 6h cloud
            if tk_cross_bull or close[i] >= min(senkou_a[i], senkou_b[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals