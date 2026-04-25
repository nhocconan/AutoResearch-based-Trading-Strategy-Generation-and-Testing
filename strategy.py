#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_VolumeBreakout
Hypothesis: Trade Ichimoku TK cross (Tenkan/Kijun) on 6h only when price is above/below 1d cloud (trend filter) AND volume breaks out (>2x 20-period MA). 
In bull markets: long when price > cloud + TK cross up + volume breakout. 
In bear markets: short when price < cloud + TK cross down + volume breakout. 
Cloud filter ensures we trade with higher timeframe trend, reducing whipsaws. 
Volume breakout confirms momentum. Discrete sizing 0.25 limits fee drag. 
Target 15-25 trades/year (~60-100 over 4 years).
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
    
    # Get 1d data for Ichimoku cloud (Senkou Span A/B) and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components on 1d: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52 displacement)
    # Tenkan-sen = (HH9 + LL9)/2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen = (HH26 + LL26)/2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A = (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B = (HH52 + LL52)/2 shifted 26 periods ahead
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # The actual cloud at time t is Senkou A/B from 26 periods ago
    # So we shift Senkou A/B BACK by 26 to get today's cloud
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid (no cloud yet)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Align cloud to 6h timeframe (completed 1d bar only)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_lagged)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_lagged)
    
    # TK cross on 6h: Tenkan/Kijun cross
    # Tenkan-sen (9) on 6h
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (high_9_6h + low_9_6h) / 2
    
    # Kijun-sen (26) on 6h
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (high_26_6h + low_26_6h) / 2
    
    # TK cross signals: tenkan > kijun (bullish cross), tenkan < kijun (bearish cross)
    tk_bullish = tenkan_6h > kijun_6h
    tk_bearish = tenkan_6h < kijun_6h
    
    # Price above/below cloud: price > Senkou A AND price > Senkou B (above cloud)
    # price < Senkou A AND price < Senkou B (below cloud)
    above_cloud = (close > senkou_a_aligned) & (close > senkou_b_aligned)
    below_cloud = (close < senkou_a_aligned) & (close < senkou_b_aligned)
    
    # Volume breakout: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_breakout = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52), TK cross (26), volume MA (20)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above cloud + TK bullish cross + volume breakout
            long_setup = above_cloud[i] and tk_bullish[i] and volume_breakout[i]
            # Short: price below cloud + TK bearish cross + volume breakout
            short_setup = below_cloud[i] and tk_bearish[i] and volume_breakout[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below cloud OR TK bearish cross
            if (~above_cloud[i]) or tk_bearish[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above cloud OR TK bullish cross
            if (~below_cloud[i]) or tk_bullish[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0