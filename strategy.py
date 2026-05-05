#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud (from 1d) + 6h Tenkan/Kijun cross + volume confirmation
# Long when: price > 1d Ichimoku cloud (Senkou Span A/B) AND Tenkan > Kijun (on 6h) AND volume > 1.5x 20-period MA
# Short when: price < 1d Ichimoku cloud AND Tenkan < Kijun AND volume > 1.5x 20-period MA
# Exit when: price crosses opposite cloud boundary OR Tenkan/Kijun cross reverses
# Uses Ichimoku for trend/filter (proven on 1d), TK cross for timing, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_IchimokuCloud_1dTKCross_VolumeConfirm"
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
    
    # Calculate volume confirmation on 6h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Tenkan (9-period) and Kijun (26-period) on 6h
    if len(high) >= 26 and len(low) >= 26:
        tenkan = (pd.Series(high).rolling(window=9, min_periods=9).max().values + 
                  pd.Series(low).rolling(window=9, min_periods=9).min().values) / 2.0
        kijun = (pd.Series(high).rolling(window=26, min_periods=26).max().values + 
                 pd.Series(low).rolling(window=26, min_periods=26).min().values) / 2.0
        tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
        tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    else:
        tenkan = np.full(n, np.nan)
        kijun = np.full(n, np.nan)
        tk_cross_up = np.zeros(n, dtype=bool)
        tk_cross_down = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need ~52 days for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(high_1d) >= 52 and len(low_1d) >= 52:
        # Tenkan-sen (Conversion Line): 9-period
        tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max().values + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min().values) / 2.0
        # Kijun-sen (Base Line): 26-period
        kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max().values + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min().values) / 2.0
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
        senkou_a = ((tenkan_1d + kijun_1d) / 2.0)
        # Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
        senkou_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max().values + 
                    pd.Series(low_1d).rolling(window=52, min_periods=52).min().values) / 2.0
        # Chikou Span (Lagging Span): close plotted 26 periods behind (not needed for cloud)
        
        # The cloud is between Senkou A and Senkou B
        # We need values shifted BACK by 26 periods to align with current price
        senkou_a_shifted = np.roll(senkou_a, 26)
        senkou_b_shifted = np.roll(senkou_b, 26)
        # Fill first 26 values with NaN (since they depend on future data)
        senkou_a_shifted[:26] = np.nan
        senkou_b_shifted[:26] = np.nan
        
        # Bullish: price above both Senkou lines
        # Bearish: price below both Senkou lines
        bullish_cloud = (close_1d > senkou_a_shifted) & (close_1d > senkou_b_shifted)
        bearish_cloud = (close_1d < senkou_a_shifted) & (close_1d < senkou_b_shifted)
    else:
        tenkan_1d = np.full(len(df_1d), np.nan)
        kijun_1d = np.full(len(df_1d), np.nan)
        senkou_a_shifted = np.full(len(df_1d), np.nan)
        senkou_b_shifted = np.full(len(df_1d), np.nan)
        bullish_cloud = np.zeros(len(df_1d), dtype=bool)
        bearish_cloud = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d Ichimoku cloud bias to 6h timeframe
    bullish_cloud_aligned = align_htf_to_ltf(prices, df_1d, bullish_cloud.astype(float))
    bearish_cloud_aligned = align_htf_to_ltf(prices, df_1d, bearish_cloud.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(bullish_cloud_aligned[i]) or np.isnan(bearish_cloud_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > cloud (bullish) AND TK cross up AND volume filter
            if (bullish_cloud_aligned[i] == 1.0 and 
                tk_cross_up[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < cloud (bearish) AND TK cross down AND volume filter
            elif (bearish_cloud_aligned[i] == 1.0 and 
                  tk_cross_down[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < cloud OR TK cross down
            if (bearish_cloud_aligned[i] == 1.0 or tk_cross_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > cloud OR TK cross up
            if (bullish_cloud_aligned[i] == 1.0 or tk_cross_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals