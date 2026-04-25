#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout with Weekly Trend Filter and Volume Spike
Hypothesis: Ichimoku cloud acts as dynamic support/resistance on daily timeframe.
Breakouts above/below cloud on 6h, aligned with weekly Kumo twist (Senkou A/B cross) trend,
and confirmed by volume spikes capture institutional momentum with low false signals.
Designed for 6h timeframe to target 12-37 trades/year by requiring confluence of
Ichimoku breakout, weekly Kumo trend, and volume confirmation.
Works in bull/bear regimes via Kumo twist filter and volume spike requirement.
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
    
    # Load 1d data ONCE for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Load 1w data ONCE for weekly Kumo twist trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Ichimoku components on 1d: Tenkan (9), Kijun (26), Senkou A/B (52)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    # Current Kumo (cloud) bounds: Senkou A and B (already shifted, so use current values)
    ichimoku_top = np.maximum(senkou_a.values, senkou_b.values)
    ichimoku_bottom = np.minimum(senkou_a.values, senkou_b.values)
    
    # Align Ichimoku cloud to 6h
    ichimoku_top_aligned = align_htf_to_ltf(prices, df_1d, ichimoku_top)
    ichimoku_bottom_aligned = align_htf_to_ltf(prices, df_1d, ichimoku_bottom)
    
    # Weekly Kumo twist trend: Senkou A/B cross on 1w indicates trend change
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    # Senkou A on 1w: (Tenkan_w + Kijun_w)/2 shifted 26
    tenkan_w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    kijun_w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
               pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    senkou_a_w = ((tenkan_w + kijun_w) / 2).shift(26)
    # Senkou B on 1w: (52-period high + 52-period low)/2 shifted 26
    senkou_b_w = ((pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                   pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    # Kumo twist: Senkou A > Senkou B = bullish twist, Senkou A < Senkou B = bearish twist
    kumo_twist_bullish = senkou_a_w.values > senkou_b_w.values
    kumo_twist_bearish = senkou_a_w.values < senkou_b_w.values
    
    # Align weekly Kumo twist to 6h
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_1w, kumo_twist_bullish.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_1w, kumo_twist_bearish.astype(float))
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(52, 20)  # Ichimoku 52-period, volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ichimoku_top_aligned[i]) or np.isnan(ichimoku_bottom_aligned[i]) or 
            np.isnan(kumo_twist_bullish_aligned[i]) or np.isnan(kumo_twist_bearish_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter from weekly Kumo twist
        bullish_bias = kumo_twist_bullish_aligned[i] > 0.5
        bearish_bias = kumo_twist_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Ichimoku breakout + Kumo twist + volume
            # Long: price breaks above Ichimoku top cloud AND bullish Kumo twist AND volume spike
            long_entry = (curr_high > ichimoku_top_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below Ichimoku bottom cloud AND bearish Kumo twist AND volume spike
            short_entry = (curr_low < ichimoku_bottom_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Ichimoku bottom cloud OR loss of bullish Kumo twist
            if (curr_low < ichimoku_bottom_aligned[i]) or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Ichimoku top cloud OR loss of bearish Kumo twist
            if (curr_high > ichimoku_top_aligned[i]) or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyKumoTwist_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0