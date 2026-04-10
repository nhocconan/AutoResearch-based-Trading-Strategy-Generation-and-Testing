#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter
# - Primary: 6h timeframe for balance of trade frequency and noise reduction
# - HTF: 1d for trend direction (price above/below 1d Kumo cloud)
# - Long: 6h Tenkan-sen crosses above Kijun-sen AND price > 1d Senkou Span A/B (bullish cloud)
# - Short: 6h Tenkan-sen crosses below Kijun-sen AND price < 1d Senkou Span A/B (bearish cloud)
# - Exit: Tenkan-sen/Kijun-sen cross in opposite direction
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 6h sweet spot
# - Works in bull/bear: Ichimoku captures trends in bull markets and avoids false signals in ranging markets via cloud filter

name = "6h_1d_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for Ichimoku calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d data for cloud calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Calculate 1d Kumo cloud (Senkou Span A/B from 1d data)
    # 1d Tenkan-sen
    high_1d_tenkan = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_1d_tenkan = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_1d_tenkan + low_1d_tenkan) / 2
    
    # 1d Kijun-sen
    high_1d_kijun = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_1d_kijun = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_1d_kijun + low_1d_kijun) / 2
    
    # 1d Senkou Span A
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # 1d Senkou Span B
    high_1d_senkou_b = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_1d_senkou_b = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_1d_senkou_b + low_1d_senkou_b) / 2
    
    # Align 1d cloud to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Determine cloud boundaries (top and bottom of cloud)
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period (max of Ichimoku periods)
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku signals
        tenkan_cross_above = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tenkan_cross_below = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Price relative to 1d cloud
        price_above_cloud = close_6h[i] > cloud_top[i]
        price_below_cloud = close_6h[i] < cloud_bottom[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Tenkan crosses above Kijun AND price above 1d cloud
            if tenkan_cross_above and price_above_cloud:
                position = 1
                signals[i] = 0.25
            # Short entry: Tenkan crosses below Kijun AND price below 1d cloud
            elif tenkan_cross_below and price_below_cloud:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Tenkan/Kijun cross in opposite direction
            if position == 1:  # Long position
                if tenkan_cross_below:  # Cross below = exit long
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if tenkan_cross_above:  # Cross above = exit short
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals