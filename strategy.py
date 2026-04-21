#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeFilter
Hypothesis: Ichimoku cloud breakouts on 6h filtered by 1d EMA50 trend and volume spike (>1.8x 20-period average).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
Ichimoku provides dynamic support/resistance via cloud, effective in both bull/bear regimes via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 6h OHLC for Ichimoku calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_sen = ((period9_high + period9_low) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_sen = ((period26_high + period26_low) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # For alignment, we'll use current values without shift for cloud calculation
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current cloud boundaries (using unshifted Senkou spans for current price action)
    senkou_a_current = senkou_a
    senkou_b_current = senkou_b
    upper_cloud = np.maximum(senkou_a_current, senkou_b_current)
    lower_cloud = np.minimum(senkou_a_current, senkou_b_current)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume filter: current volume > 1.8x 20-period average ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):  # Warmup for Ichimoku (52 periods)
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) 
            or np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i])
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume filter
            vol_filter = volume[i] > 1.8 * vol_ma[i]
            
            # Long conditions: price above cloud, TK cross bullish, 1d uptrend
            price_above_cloud = price > upper_cloud[i]
            tk_bullish = tenkan_sen[i] > kijun_sen[i]
            trend_up = price > ema_50_1d_aligned[i]
            
            # Short conditions: price below cloud, TK cross bearish, 1d downtrend
            price_below_cloud = price < lower_cloud[i]
            tk_bearish = tenkan_sen[i] < kijun_sen[i]
            trend_down = price < ema_50_1d_aligned[i]
            
            # Entry logic
            if price_above_cloud and tk_bullish and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif price_below_cloud and tk_bearish and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit: price closes below cloud or TK cross bearish
            if price < lower_cloud[i] or tenkan_sen[i] < kijun_sen[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above cloud or TK cross bullish
            if price > upper_cloud[i] or tenkan_sen[i] > kijun_sen[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0