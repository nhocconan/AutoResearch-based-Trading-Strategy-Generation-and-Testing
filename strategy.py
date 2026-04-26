#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter_v1
Hypothesis: Use Ichimoku cloud twist (Senkou Span A/B cross) on 6h as primary trend change signal, confirmed by 1d EMA50 trend and volume spike. Works in both bull and bear markets by only taking trades in direction of higher timeframe trend. Targets 12-25 trades/year to minimize fee drag.
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
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Kumo twist: Senkou A crosses above/below Senkou B
    # We need to align these to current time (no shift for signal generation)
    # Since Senkou spans are plotted 26 periods ahead, we compare current Senkou A/B
    # ABOVE cloud: price > max(Senkou A, Senkou B)
    # BELOW cloud: price < min(Senkou A, Senkou B)
    # TWIST: Senkou A crosses Senkou B
    senkou_a_shifted = np.roll(senkou_a, 26)  # shift back to align with current price
    senkou_b_shifted = np.roll(senkou_b, 26)
    
    # Handle NaN from roll
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 for Ichimoku, 20 for volume, 50 for 1d EMA
    start_idx = max(52, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size
        
        # Kumo twist signals
        # Bullish twist: Senkou A crosses above Senkou B
        bullish_twist = (senkou_a_shifted[i] > senkou_b_shifted[i]) and (senkou_a_shifted[i-1] <= senkou_b_shifted[i-1])
        # Bearish twist: Senkou A crosses below Senkou B
        bearish_twist = (senkou_a_shifted[i] < senkou_b_shifted[i]) and (senkou_a_shifted[i-1] >= senkou_b_shifted[i-1])
        
        # Price above/below cloud
        above_cloud = close_val > max(senkou_a_shifted[i], senkou_b_shifted[i])
        below_cloud = close_val < min(senkou_a_shifted[i], senkou_b_shifted[i])
        
        if position == 0:
            # Flat - look for twist with trend and volume confirmation
            # Long: bullish twist + price above cloud + 1d EMA50 uptrend + volume spike
            long_entry = bullish_twist and above_cloud and (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and volume_spike[i]
            # Short: bearish twist + price below cloud + 1d EMA50 downtrend + volume spike
            short_entry = bearish_twist and below_cloud and (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price closes below cloud or twist reverses
            if below_cloud or bearish_twist:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price closes above cloud or twist reverses
            if above_cloud or bullish_twist:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0