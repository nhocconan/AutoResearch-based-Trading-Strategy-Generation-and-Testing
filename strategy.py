#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_breakout_v1
# Strategy: Ichimoku cloud breakout with daily trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Price breaking above/below Ichimoku cloud with daily trend alignment captures
# strong momentum moves. Cloud acts as dynamic support/resistance, reducing whipsaws.
# Works in bull by catching breakouts above cloud in uptrend, in bear by catching
# breakdowns below cloud in downtrend. Volume confirms institutional participation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back (not used for signals)
    
    # Align Ichimoku components to 6h timeframe (with proper delay for leading spans)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)  # Leading span needs delay
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)  # Leading span needs delay
    
    # Daily trend filter: price above/below cloud
    # Green cloud (bullish): Senkou A > Senkou B
    # Red cloud (bearish): Senkou A < Senkou B
    green_cloud = senkou_a_6h > senkou_b_6h
    red_cloud = senkou_a_6h < senkou_b_6h
    
    # Price relative to cloud
    above_cloud = (close > senkou_a_6h) & (close > senkou_b_6h)
    below_cloud = (close < senkou_a_6h) & (close < senkou_b_6h)
    
    # Volume confirmation: 20-period volume average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Senkou B calculation period
        # Skip if any required data is invalid
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Bullish breakout: price breaks above cloud in bullish trend
        bull_breakout = (above_cloud[i] and 
                        not above_cloud[i-1] and  # Just broke above
                        green_cloud[i] and        # Bullish cloud
                        vol_spike[i])             # Volume confirmation
        
        # Bearish breakdown: price breaks below cloud in bearish trend
        bear_breakdown = (below_cloud[i] and 
                         not below_cloud[i-1] and  # Just broke below
                         red_cloud[i] and          # Bearish cloud
                         vol_spike[i])             # Volume confirmation
        
        # Exit conditions: price returns to cloud or opposite signal
        exit_long = position == 1 and (not above_cloud[i] or bear_breakdown)
        exit_short = position == -1 and (not below_cloud[i] or bull_breakout)
        
        # Trading logic
        if bull_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_breakdown and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals