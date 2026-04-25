#!/usr/bin/env python3
"""
6h Ichimoku Cloud + TK Cross + 1d Trend Filter
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. TK line crosses above/below cloud with 1d EMA50 trend filter capture momentum in both bull/bear markets. Cloud filter prevents whipsaws in ranging markets. Volume confirmation adds institutional validation. Designed for 6h timeframe to balance trade frequency and signal quality.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
    # Not used in signals as it requires future data
    
    # Cloud top and bottom (for current price)
    # Cloud is plotted 26 periods ahead, so we use values shifted back by 26 for current comparison
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # Fill first 26 values with NaN (no cloud data)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_above = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_below = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Volume spike: current volume > 2.0 * 20-period average
    if n >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    else:
        vol_ma_20 = np.full(n, np.mean(volume))
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52) and EMA50 (50)
    start_idx = max(52, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_volume_spike = volume_spike[i]
        ema_trend = ema_50_aligned[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        
        # Determine if price is above/below cloud
        above_cloud = curr_close > cloud_top_val
        below_cloud = curr_close < cloud_bottom_val
        in_cloud = ~(above_cloud | below_cloud)
        
        # Exit conditions: TK cross in opposite direction or price re-enters cloud
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish TK cross or price re-enters cloud
                if tk_cross_below[i] or in_cloud:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish TK cross or price re-enters cloud
                if tk_cross_above[i] or in_cloud:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: TK cross + price above/below cloud + trend alignment + volume spike
        if position == 0:
            # Long: bullish TK cross AND price above cloud AND price above 1d EMA50
            long_condition = tk_cross_above[i] and above_cloud and (curr_close > ema_trend) and curr_volume_spike
            # Short: bearish TK cross AND price below cloud AND price below 1d EMA50
            short_condition = tk_cross_below[i] and below_cloud and (curr_close < ema_trend) and curr_volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TKCross_CloudFilter_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0