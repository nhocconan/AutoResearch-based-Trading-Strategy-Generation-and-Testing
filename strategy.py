#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku cloud with TK cross and volume confirmation
# Ichimoku cloud from 1d provides strong support/resistance aligned with 6h timeframe
# Tenkan-Kijun cross signals momentum shifts, cloud acts as dynamic filter
# Volume confirmation (current 6h volume > 1.8x 30-period average) ensures breakout validity
# Works in bull/bear: price respects Ichimoku structure, TK cross catches reversals
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_ichimoku_tk_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Ichimoku calculations
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max()
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b = ((high_senkou_b + low_senkou_b) / 2.0)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    # Determine cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Pre-compute volume confirmation (30-period average for 6h)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.8x average 6h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_30[i]
        
        if position == 1:  # Long position
            # Exit when price closes below cloud or TK cross turns bearish
            if close[i] < cloud_bottom[i] or (tenkan_aligned[i] < kijun_aligned[i] and close[i] < tenkan_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price closes above cloud or TK cross turns bullish
            if close[i] > cloud_top[i] or (tenkan_aligned[i] > kijun_aligned[i] and close[i] > tenkan_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter on TK cross with volume confirmation and price outside cloud
            if volume_confirmed:
                # Bullish TK cross: Tenkan crosses above Kijun
                if tenkan_aligned[i] > kijun_aligned[i] and close[i] > cloud_top[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish TK cross: Tenkan crosses below Kijun
                elif tenkan_aligned[i] < kijun_aligned[i] and close[i] < cloud_bottom[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals