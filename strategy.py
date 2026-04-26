#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v1
Hypothesis: Ichimoku TK cross with cloud filter from 1d on 6h timeframe. 
Long when TK cross bullish + price above cloud + bullish weekly bias. 
Short when TK cross bearish + price below cloud + bearish weekly bias.
Uses volume confirmation (1.5x median) to avoid whipsaws. 
Designed for ~15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
Works in bull via trend following and in bear via short signals with cloud filter.
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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need enough for Ichimoku components
        return np.zeros(n)
    
    # Get 1w data for weekly bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for signals as it requires future data
    
    # Weekly bias from 1w: price > EMA21 for bullish, < EMA21 for bearish
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align all HTF indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume confirmation: 1.5x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku components (26 for base line) + weekly EMA + volume
    start_idx = max(26, 21, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_median[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_21_1w_val = ema_21_1w_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # TK cross signals
        tk_bullish = tenkan_val > kijun_val
        tk_bearish = tenkan_val < kijun_val
        
        # Price relative to cloud
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        # Weekly bias
        weekly_bullish = close_val > ema_21_1w_val
        weekly_bearish = close_val < ema_21_1w_val
        
        # Volume confirmation
        volume_confirmed = volume_val > 1.5 * vol_median_val
        
        if position == 0:
            # Long: bullish TK cross + price above cloud + bullish weekly + volume
            long_signal = tk_bullish and price_above_cloud and weekly_bullish and volume_confirmed
            
            # Short: bearish TK cross + price below cloud + bearish weekly + volume
            short_signal = tk_bearish and price_below_cloud and weekly_bearish and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions: bearish TK cross OR price drops below cloud OR weekly turns bearish
            if tk_bearish or not price_above_cloud or not weekly_bullish:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions: bullish TK cross OR price rises above cloud OR weekly turns bullish
            if tk_bullish or not price_below_cloud or not weekly_bearish:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_v1"
timeframe = "6h"
leverage = 1.0