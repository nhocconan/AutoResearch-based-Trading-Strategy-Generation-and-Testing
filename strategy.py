#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_1dVolume_v1
Hypothesis: Ichimoku cloud breakout on 6h with weekly trend filter (price > weekly EMA50) and 1d volume confirmation (>1.8x median). Uses cloud as dynamic support/resistance and TK cross for momentum. Weekly EMA50 ensures we only trade with higher timeframe trend, reducing false breaks in chop. Designed for BTC/ETH with ~30-60 trades/year to avoid fee drag. Works in bull/bear by only taking breaks in direction of weekly trend.
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
    
    # Get 1d data for volume median
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
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
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used for signals to avoid look-ahead
    
    # The cloud is between Senkou Span A and Senkou Span B
    # Upper cloud = max(Senkou A, Senkou B)
    # Lower cloud = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Volume spike filter: volume > 1.8x median volume (30-period) for high conviction
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    upper_cloud_aligned = align_htf_to_ltf(prices, df_1d, upper_cloud)
    lower_cloud_aligned = align_htf_to_ltf(prices, df_1d, lower_cloud)
    tk_cross_up_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_up.astype(float))
    tk_cross_down_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_down.astype(float))
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku (52), weekly EMA (50), volume median (30)
    start_idx = max(52, 50, 30) + 26  # +26 for cloud shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_cloud_aligned[i]) or
            np.isnan(lower_cloud_aligned[i]) or
            np.isnan(tk_cross_up_aligned[i]) or
            np.isnan(tk_cross_down_aligned[i]) or
            np.isnan(vol_median_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median_aligned[i]
        upper_cloud_val = upper_cloud_aligned[i]
        lower_cloud_val = lower_cloud_aligned[i]
        tk_cross_up_val = tk_cross_up_aligned[i] > 0.5
        tk_cross_down_val = tk_cross_down_aligned[i] > 0.5
        
        # Trend filter: price > weekly EMA50 (uptrend) or < weekly EMA50 (downtrend)
        uptrend = close_val > ema_50_1w_val
        downtrend = close_val < ema_50_1w_val
        
        # Volume confirmation
        volume_ok = volume_val > 1.8 * vol_median_val
        
        if position == 0:
            # Long: price breaks above cloud with TK cross up, volume, and uptrend
            long_signal = (close_val > upper_cloud_val) and \
                          tk_cross_up_val and \
                          volume_ok and \
                          uptrend
            
            # Short: price breaks below cloud with TK cross down, volume, and downtrend
            short_signal = (close_val < lower_cloud_val) and \
                           tk_cross_down_val and \
                           volume_ok and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long: exit if price re-enters cloud or TK cross down
            signals[i] = 0.25
            if close_val < lower_cloud_val or tk_cross_down_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short: exit if price re-enters cloud or TK cross up
            signals[i] = -0.25
            if close_val > upper_cloud_val or tk_cross_up_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_1dVolume_v1"
timeframe = "6h"
leverage = 1.0