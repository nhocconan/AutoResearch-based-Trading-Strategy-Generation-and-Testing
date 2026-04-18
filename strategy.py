#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_Volume
- Ichimoku cloud from 1d timeframe for trend/filter
- TK cross from 6h price for entry signal
- Volume confirmation on 6h
- Long: Price breaks above cloud + TK cross bullish + volume > 1.5x 20-period average
- Short: Price breaks below cloud + TK cross bearish + volume > 1.5x 20-period average
- Exit: Opposite TK cross or price returns to cloud center
- Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
- Works in bull markets (trend continuation) and bear markets (trend continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components for daily timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used in signals as it requires future data
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    # TK cross on 6h timeframe
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2.0
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2.0
    
    tk_cross_bullish = tenkan_6h > kijun_6h
    tk_cross_bearish = tenkan_6h < kijun_6h
    
    # Volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # need enough for Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(tk_cross_bullish[i]) or np.isnan(tk_cross_bearish[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top_aligned[i]
        price_below_cloud = close[i] < cloud_bottom_aligned[i]
        price_in_cloud = (close[i] >= cloud_bottom_aligned[i]) and (close[i] <= cloud_top_aligned[i])
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price above cloud + TK bullish + volume
            if price_above_cloud and tk_cross_bullish[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK bearish + volume
            elif price_below_cloud and tk_cross_bearish[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to cloud or TK bearish
            if price_in_cloud or not tk_cross_bullish[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to cloud or TK bullish
            if price_in_cloud or not tk_cross_bearish[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_Volume"
timeframe = "6h"
leverage = 1.0