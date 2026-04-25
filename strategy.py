#!/usr/bin/env python3
"""
6h_IchiCloud_Alligator_Trend
Hypothesis: 6h timeframe with 1d Ichimoku cloud filter + Williams Alligator (13,8,5) smoothed MAs.
Long when price > cloud (Senkou Span A/B), Tenkan > Kijun, and Alligator bullish (Lips > Teeth > Jaw).
Short when price < cloud, Tenkan < Kijun, and Alligator bearish (Lips < Teeth < Jaw).
Uses volume confirmation and discrete sizing (0.25) to limit trades to 12-37/year.
Works in bull/bear via trend-following with cloud as dynamic support/resistance.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Ichimoku and Alligator (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku calculations (9,26,52 periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_period = 9
    max_high_9 = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_period = 26
    max_high_26 = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b_period = 52
    max_high_52 = pd.Series(high_1d).rolling(window=senkou_b_period, min_periods=senkou_b_period).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=senkou_b_period, min_periods=senkou_b_period).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Williams Alligator (13,8,5) with smoothing (8,5,3)
    jaw_period = 13    # Alligator's Jaw (slowest)
    teeth_period = 8   # Alligator's Teeth
    lips_period = 5    # Alligator's Lips (fastest)
    
    # Smoothed moving averages (using SMA for simplicity, but could use SMMA)
    jaw = pd.Series(close_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(close_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(close_1d).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Align all 1d indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)  # Leading span needs extra delay
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)  # Leading span needs extra delay
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52+26=78) and Alligator (13)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Ichimoku cloud: price above/below cloud
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Tenkan/Kijun cross
        tenkan_gt_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_lt_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        # Alligator alignment: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        alligator_bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        alligator_bearish = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Look for entry signals with all conditions aligned
            # Long: price above cloud, Tenkan > Kijun, Alligator bullish, volume confirmation
            long_signal = price_above_cloud and tenkan_gt_kijun and alligator_bullish and volume_confirm[i]
            # Short: price below cloud, Tenkan < Kijun, Alligator bearish, volume confirmation
            short_signal = price_below_cloud and tenkan_lt_kijun and alligator_bearish and volume_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below cloud or Alligator turns bearish
            if curr_close < cloud_bottom or not alligator_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above cloud or Alligator turns bullish
            if curr_close > cloud_top or not alligator_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchiCloud_Alligator_Trend"
timeframe = "6h"
leverage = 1.0