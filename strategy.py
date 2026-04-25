#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v2
Hypothesis: Use Ichimoku Tenkan/Kijun cross with daily cloud filter on 6h timeframe.
Long when Tenkan crosses above Kijun AND price is above daily cloud (Senkou Span A/B).
Short when Tenkan crosses below Kijun AND price is below daily cloud.
Requires volume > 1.3x 20-period average for confirmation.
Exit on opposite TK cross or when price re-enters the cloud.
Position size: 0.25 to limit drawdown.
Target: 50-150 total trades over 4 years = 12-37/year.
Works in bull (bullish TK cross above cloud) and bear (bearish TK cross below cloud) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need sufficient data for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 20-period average volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52) and volume MA (20)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Price relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        # TK cross signals
        tk_bullish_cross = (tenkan_aligned[i] > kijun_aligned[i]) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
        tk_bearish_cross = (tenkan_aligned[i] < kijun_aligned[i]) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
        
        if position == 0:
            # Long setup: bullish TK cross + price above cloud + volume confirmation
            long_setup = tk_bullish_cross and price_above_cloud and volume_confirm
            
            # Short setup: bearish TK cross + price below cloud + volume confirmation
            short_setup = tk_bearish_cross and price_below_cloud and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: bearish TK cross OR price re-enters cloud (below upper cloud)
            if tk_bearish_cross or (close[i] <= upper_cloud):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: bullish TK cross OR price re-enters cloud (above lower cloud)
            if tk_bullish_cross or (close[i] >= lower_cloud):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_v2"
timeframe = "6h"
leverage = 1.0