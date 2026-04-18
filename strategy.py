#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1D
Hypothesis: On 6h chart, enter long when Tenkan-sen > Kijun-sen and price > Kumo cloud (Senkou Span A/B) derived from 1d data, with volume confirmation; enter short on opposite conditions. Uses Ichimoku cloud as dynamic support/resistance and trend filter. The 1d Ichimoku provides higher-timeframe trend bias, reducing whipsaws in both bull and bear markets. Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # Need at least 26*2 for Ichimoku calculation
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation (higher timeframe trend)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h timeframe (wait for 1d bar to close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume filter: >1.5x 20-period average on 6h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 52  # Warmup for Ichimoku calculations
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        vol_ok = volume_filter[i]
        
        # Determine cloud boundaries and trend
        upper_cloud = max(senkou_a, senkou_b)
        lower_cloud = min(senkou_a, senkou_b)
        cloud_bullish = senkou_a > senkou_b  # Cloud is bullish when Senkou A > Senkou B
        
        if position == 0:
            # Long: TK bullish cross, price above cloud, bullish cloud, with volume
            if (tenkan > kijun and price > upper_cloud and cloud_bullish and vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: TK bearish cross, price below cloud, bearish cloud, with volume
            elif (tenkan < kijun and price < lower_cloud and not cloud_bullish and vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TK bearish cross or price drops below cloud
            if (tenkan < kijun or price < lower_cloud):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TK bullish cross or price rises above cloud
            if (tenkan > kijun or price > upper_cloud):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1D"
timeframe = "6h"
leverage = 1.0