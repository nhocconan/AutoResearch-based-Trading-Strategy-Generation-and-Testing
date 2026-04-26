#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeFilter
Hypothesis: On 6h timeframe, enter long when price breaks above Ichimoku cloud (Senkou Span A/B) and Tenkan/Kijun cross is bullish, aligned with 1d EMA50 trend and volume > 1.5x 20-bar MA. Enter short on mirror conditions. Uses discrete sizing (0.25) to limit fee drag. Ichimoku provides dynamic support/resistance and trend confirmation, working in both bull (cloud as support) and bear (cloud as resistance) markets via trend alignment filter.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for entry)
    
    # The actual cloud boundaries at current price are Senkou A/B from 26 periods ago
    # So we need to shift Senkou A/B BACK by 26 to align with current price
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # Tenkan/Kijun cross
    tk_cross = tenkan - kijun  # >0 bullish, <0 bearish
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (52 for Senkou B, 26 for shift, 20 for vol)
    start_idx = max(52 + 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tk_cross[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        tk_cross_val = tk_cross[i]
        vol_spike = volume_spike[i]
        ema_50_val = ema_50_1d_aligned[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Entry conditions
        # Long: price above cloud, TK cross bullish, volume spike, 1d trend bullish
        long_entry = (close_val > cloud_top_val) and (tk_cross_val > 0) and vol_spike and bullish_1d
        # Short: price below cloud, TK cross bearish, volume spike, 1d trend bearish
        short_entry = (close_val < cloud_bottom_val) and (tk_cross_val < 0) and vol_spike and bearish_1d
        
        # Exit conditions: price returns inside cloud or TK cross reverses or 1d trend changes
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < cloud_top_val or tk_cross_val <= 0 or not bullish_1d):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > cloud_bottom_val or tk_cross_val >= 0 or not bearish_1d):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0