#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku TK Cross + 1d Cloud Filter + Volume Spike
- Ichimoku TK Cross (Tenkan/Kijun) on 6h captures momentum shifts with proven edge
- 1d Cloud (Senkou Span A/B) acts as higher timeframe trend filter: only long when price above cloud, short when below
- Volume spike (2.0x 20-period MA) confirms institutional participation
- Discrete position sizing (0.25) minimizes fee churn
- Target: 12-30 trades/year per symbol (~50-120 total over 4 years)
- Works in bull markets (TK cross up in bullish cloud) and bear markets (TK cross down in bearish cloud)
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
    
    # Get 1d data for cloud filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 6h data for primary timeframe (Ichimoku, volume)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Calculate 1d Cloud (HTF)
    # Senkou Span A: (9-period high + 9-period low)/2 on 1d, shifted 26 periods ahead
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    senkou_a_1d = ((period9_high_1d + period9_low_1d) / 2.0)
    
    # Senkou Span B: (52-period high + 52-period low)/2 on 1d, shifted 26 periods ahead
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((period52_high_1d + period52_low_1d) / 2.0)
    
    # Calculate volume average (20-period) on 6h
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        senkou_a_1d_val = senkou_a_1d_aligned[i]
        senkou_b_1d_val = senkou_b_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Determine cloud boundaries and trend
        cloud_top = max(senkou_a_1d_val, senkou_b_1d_val)
        cloud_bottom = min(senkou_a_1d_val, senkou_b_1d_val)
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        if position == 0:
            # Look for TK cross with volume confirmation and cloud filter
            # Bullish TK cross: Tenkan crosses above Kijun
            bullish_cross = tenkan_val > kijun_val and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            # Bearish TK cross: Tenkan crosses below Kijun
            bearish_cross = tenkan_val < kijun_val and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            
            # Long: bullish TK cross + price above cloud + volume spike
            if bullish_cross and price_above_cloud and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross + price below cloud + volume spike
            elif bearish_cross and price_below_cloud and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish TK cross OR price drops below cloud
            bearish_cross = tenkan_val < kijun_val and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            if bearish_cross or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish TK cross OR price rises above cloud
            bullish_cross = tenkan_val > kijun_val and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            if bullish_cross or price > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuTKCross_1dCloudFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0