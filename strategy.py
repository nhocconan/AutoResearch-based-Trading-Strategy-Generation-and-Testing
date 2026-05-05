#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d Weekly Cloud filter and Volume Spike
# Long when price breaks above 6h Tenkan/Kijun cross AND price > 1d Weekly Kumo (cloud top) AND volume spike
# Short when price breaks below 6h Tenkan/Kijun cross AND price < 1d Weekly Kumo (cloud bottom) AND volume spike
# Ichimoku provides dynamic support/resistance with trend/momentum signals via TK cross
# 1d Weekly Cloud (from higher timeframe) filters for alignment with major trend, reducing whipsaw
# Volume spike (2.0x 20-bar MA) confirms breakout strength
# Works in bull (trend + breakouts above cloud) and bear (breakdowns below cloud with volume)
# Timeframe: 6h (as required)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing high-probability breaks

name = "6h_Ichimoku_WeeklyCloud_TKCross_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data ONCE before loop for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need at least 52 for Ichimoku (26*2)
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for Weekly Cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 6h Ichimoku components
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2.0
    
    # Align Ichimoku components to 6h timeframe (no shift needed as Ichimoku uses current bar data)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Calculate 1d Weekly Cloud (using 1d data to derive weekly levels)
    # Weekly high/low from 1d data (approximate weekly by 5-day period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly Tenkan (5-period): (5-day high + low)/2
    period_weekly_tenkan = 5
    max_high_weekly_tenkan = pd.Series(high_1d).rolling(window=period_weekly_tenkan, min_periods=period_weekly_tenkan).max().values
    min_low_weekly_tenkan = pd.Series(low_1d).rolling(window=period_weekly_tenkan, min_periods=period_weekly_tenkan).min().values
    weekly_tenkan = (max_high_weekly_tenkan + min_low_weekly_tenkan) / 2.0
    
    # Weekly Kijun (26-period approximate): (26-day high + low)/2 - using 26 for weekly (~1 month)
    period_weekly_kijun = 26
    max_high_weekly_kijun = pd.Series(high_1d).rolling(window=period_weekly_kijun, min_periods=period_weekly_kijun).max().values
    min_low_weekly_kijun = pd.Series(low_1d).rolling(window=period_weekly_kijun, min_periods=period_weekly_kijun).min().values
    weekly_kijun = (max_high_weekly_kijun + min_low_weekly_kijun) / 2.0
    
    # Weekly Senkou Span A: (Weekly Tenkan + Weekly Kijun)/2
    weekly_senkou_a = (weekly_tenkan + weekly_kijun) / 2.0
    
    # Weekly Senkou Span B: (52-day high + low)/2
    period_weekly_senkou_b = 52
    max_high_weekly_senkou_b = pd.Series(high_1d).rolling(window=period_weekly_senkou_b, min_periods=period_weekly_senkou_b).max().values
    min_low_weekly_senkou_b = pd.Series(low_1d).rolling(window=period_weekly_senkou_b, min_periods=period_weekly_senkou_b).min().values
    weekly_senkou_b = (max_high_weekly_senkou_b + min_low_weekly_senkou_b) / 2.0
    
    # Weekly Cloud Top (Senkou Span A) and Bottom (Senkou Span B)
    weekly_kumo_top = np.maximum(weekly_senkou_a, weekly_senkou_b)
    weekly_kumo_bottom = np.minimum(weekly_senkou_a, weekly_senkou_b)
    
    # Align Weekly Cloud to 6h timeframe
    weekly_kumo_top_aligned = align_htf_to_ltf(prices, df_1d, weekly_kumo_top)
    weekly_kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, weekly_kumo_bottom)
    
    # Volume confirmation on 6h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any value is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(weekly_kumo_top_aligned[i]) or np.isnan(weekly_kumo_bottom_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross bullish (Tenkan > Kijun) AND price > Weekly Cloud Top AND volume spike
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                close[i] > weekly_kumo_top_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish (Tenkan < Kijun) AND price < Weekly Cloud Bottom AND volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  close[i] < weekly_kumo_bottom_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross bearish OR price falls below Weekly Cloud Bottom
            if tenkan_aligned[i] < kijun_aligned[i] or close[i] < weekly_kumo_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above Weekly Cloud Top
            if tenkan_aligned[i] > kijun_aligned[i] or close[i] > weekly_kumo_top_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals