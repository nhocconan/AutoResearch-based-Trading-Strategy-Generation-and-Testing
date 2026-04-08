#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1-day trend filter and volume confirmation
Hypothesis: Ichimoku (Tenkan/Kijun cross + price vs Cloud) on 6h timeframe,
filtered by 1-day Senkou Span trend (bullish/bearish cloud) and volume > 1.5x average,
captures strong trends while avoiding whipsaws in ranging markets.
Designed for ~15-25 trades/year to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku on 6h: Tenkan (9), Kijun (26), Senkou Span B (52)
    # Tenkan-sen: (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = ((high_9 + low_9) / 2).values
    
    # Kijun-sen: (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = ((high_26 + low_26) / 2).values
    
    # Senkou Span A: (Tenkan + Kijun)/2 shifted 26 periods ahead
    # Senkou Span B: (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2)
    
    # For cloud calculation, we need current Senkou Span values (no shift for cloud plot)
    # But for trend filter, we use 1-day Senkou Span
    
    # 1-day Ichimoku for trend filter
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_1d = ((high_9_1d + low_9_1d) / 2).values
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_1d = ((high_26_1d + low_26_1d) / 2).values
    
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    senkou_b_1d = ((high_52_1d + low_52_1d) / 2)
    
    # 1-day trend: bullish if Senkou A > Senkou B, bearish if Senkou A < Senkou B
    trend_bullish = senkou_a_1d > senkou_b_1d
    trend_bearish = senkou_a_1d < senkou_b_1d
    
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish)
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Need 52 periods for Senkou Span B
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud color and position relative to cloud
        # Green cloud (bullish): Senkou A > Senkou B
        # Red cloud (bearish): Senkou A < Senkou B
        cloud_green = senkou_a[i] > senkou_b[i]
        cloud_red = senkou_a[i] < senkou_b[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > max(senkou_a[i], senkou_b[i])
        price_below_cloud = close[i] < min(senkou_a[i], senkou_b[i])
        
        if position == 1:  # Long position
            # Exit: Tenkan crosses below Kijun OR price enters cloud OR trend turns bearish
            if (tenkan[i] <= kijun[i] or 
                not price_above_cloud or 
                trend_bearish_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan crosses above Kijun OR price enters cloud OR trend turns bullish
            if (tenkan[i] >= kijun[i] or 
                not price_below_cloud or 
                trend_bullish_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Tenkan crosses above Kijun + price above cloud + bullish trend + volume spike
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and  # Cross
                price_above_cloud and 
                trend_bullish_aligned[i] and
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: Tenkan crosses below Kijun + price below cloud + bearish trend + volume spike
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and  # Cross
                  price_below_cloud and 
                  trend_bearish_aligned[i] and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals