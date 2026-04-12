#!/usr/bin/env python3
"""
6h_1d_Ichimoku_Kumo_Twist_v1
Hypothesis: Trade Ichimoku Kumo (cloud) twists on 1d timeframe with 6h price confirmation.
- Kumo twist occurs when Senkou Span A and Senkou Span B cross, indicating trend change.
- Enter long when price is above cloud and TK line crosses above Kijun (bullish twist).
- Enter short when price is below cloud and TK line crosses below Kijun (bearish twist).
- Use volume confirmation (2x 20-period average) to filter false signals.
- Works in bull markets (riding trends) and bear markets (catching reversals).
- Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_Kumo_Twist_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # Need at least 52 for Ichimoku calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR ICHIMOKU ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals but needed for alignment
    
    # Align Ichimoku components to 6h timeframe (wait for daily close)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # === VOLUME FILTER ON 6H CHART ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Kumo twist: Senkou A and Senkou B cross
        # Bullish twist: Senkou A crosses above Senkou B
        # Bearish twist: Senkou A crosses below Senkou B
        bullish_twist = (senkou_a_aligned[i] > senkou_b_aligned[i] and 
                        senkou_a_aligned[i-1] <= senkou_b_aligned[i-1])
        bearish_twist = (senkou_a_aligned[i] < senkou_b_aligned[i] and 
                        senkou_a_aligned[i-1] >= senkou_b_aligned[i-1])
        
        # Price position relative to cloud
        above_cloud = (close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i])
        below_cloud = (close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i])
        
        # TK crossover (Tenkan/Kijun)
        tk_bullish = (tenkan_aligned[i] > kijun_aligned[i] and 
                     tenkan_aligned[i-1] <= kijun_aligned[i-1])
        tk_bearish = (tenkan_aligned[i] < kijun_aligned[i] and 
                     tenkan_aligned[i-1] >= kijun_aligned[i-1])
        
        # Volume confirmation
        strong_volume = volume[i] > (vol_ma[i] * 2.0)
        
        # Long: bullish twist + price above cloud + TK bullish + volume
        long_signal = (bullish_twist and above_cloud and tk_bullish and strong_volume)
        
        # Short: bearish twist + price below cloud + TK bearish + volume
        short_signal = (bearish_twist and below_cloud and tk_bearish and strong_volume)
        
        # Exit: opposite TK crossover or price crosses opposite cloud edge
        exit_long = (position == 1 and 
                    (tk_bearish or 
                     close[i] < senkou_a_aligned[i] or close[i] < senkou_b_aligned[i]))
        exit_short = (position == -1 and 
                     (tk_bullish or 
                      close[i] > senkou_a_aligned[i] or close[i] > senkou_b_aligned[i]))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals