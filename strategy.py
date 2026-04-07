#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Ichimoku Cloud + 1D Trend + Volume Filter
# Hypothesis: Ichimoku cloud from daily timeframe provides dynamic support/resistance
# and trend direction. Tenkan/Kijun cross on 6h provides entry timing, filtered by
# daily cloud color (bullish/bearish) and volume surge. Works in both bull/bear
# markets by adapting to trend via cloud position.
# Target: 20-30 trades/year to stay well under 300 max trades on 6h.
name = "6h_ichimoku_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Ichimoku calculations
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = df_1d['high'].rolling(window=9, min_periods=9).max()
    low_9 = df_1d['low'].rolling(window=9, min_periods=9).min()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = df_1d['high'].rolling(window=26, min_periods=26).max()
    low_26 = df_1d['low'].rolling(window=26, min_periods=26).min()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(2)  # Shifted 2 periods ahead
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = df_1d['high'].rolling(window=52, min_periods=52).max()
    low_52 = df_1d['low'].rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2).shift(2)  # Shifted 2 periods ahead
    
    # Cloud is bullish when Senkou A > Senkou B
    cloud_bullish = senkou_a > senkou_b
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    cloud_bullish_6h = align_htf_to_ltf(prices, df_1d, cloud_bullish.values.astype(float))
    
    # Volume filter: current volume > 1.8x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if required data not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(cloud_bullish_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries
        top_cloud = max(senkou_a_6h[i], senkou_b_6h[i])
        bottom_cloud = min(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 1:  # Long position
            # Exit: price falls below cloud OR Tenkan/Kijun death cross
            if close[i] < bottom_cloud or (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price rises above cloud OR Tenkan/Kijun golden cross
            if close[i] > top_cloud or (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Golden cross (bullish): Tenkan crosses above Kijun
                if tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]:
                    # Only take if price is above cloud (bullish regime)
                    if close[i] > top_cloud:
                        position = 1
                        signals[i] = 0.25
                # Death cross (bearish): Tenkan crosses below Kijun
                elif tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]:
                    # Only take if price is below cloud (bearish regime)
                    if close[i] < bottom_cloud:
                        position = -1
                        signals[i] = -0.25
    
    return signals