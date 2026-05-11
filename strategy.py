#!/usr/bin/env python3
"""
1d_Weekly_Ichimoku_Cloud_Trend_Strategy
Hypothesis: Use weekly Ichimoku Cloud for trend direction and support/resistance on daily timeframe.
Buy when price is above cloud and Tenkan > Kijun (bullish), sell when price is below cloud and Tenkan < Kijun (bearish).
Exit when price crosses back into cloud or Tenkan/Kijun cross reverses.
Works in bull markets via cloud support and in bear markets via cloud resistance.
"""

name = "1d_Weekly_Ichimoku_Cloud_Trend_Strategy"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for Ichimoku
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 52:  # Need at least 52 weeks for proper calculation
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly data
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_w).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_w).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_w).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # We don't use Chikou for signals to avoid look-ahead
    
    # Align Ichimoku components to daily timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_weekly, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_weekly, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_weekly, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_weekly, senkou_b)
    
    # Calculate cloud top and bottom
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Ichimoku (need 52 periods)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above cloud AND Tenkan > Kijun (bullish)
            if (close[i] > cloud_top[i] and tenkan_aligned[i] > kijun_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND Tenkan < Kijun (bearish)
            elif (close[i] < cloud_bottom[i] and tenkan_aligned[i] < kijun_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price falls below cloud OR Tenkan < Kijun (trend weakness)
                if (close[i] < cloud_top[i]) or (tenkan_aligned[i] < kijun_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above cloud OR Tenkan > Kijun (trend reversal)
                if (close[i] > cloud_bottom[i]) or (tenkan_aligned[i] > kijun_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals