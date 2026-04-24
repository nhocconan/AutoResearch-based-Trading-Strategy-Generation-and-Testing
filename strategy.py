#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d Weekly Pivot confirmation and volume spike.
- Ichimoku TK cross (Tenkan/Kijun) from 6h provides momentum signal.
- Cloud (Senkou Span A/B) from 6h acts as dynamic support/resistance filter.
- 1d Weekly Pivot (PP, R1, S1) provides higher timeframe structure: long only above PP, short only below PP.
- Volume spike (>2.0x 20-period average) confirms breakout validity.
- Discrete position sizing (0.25) balances return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
- Uses 1d HTF data loaded ONCE before loop per MTF rules.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Ichimoku (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    
    # Get 1d data for Weekly Pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 7:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Align Ichimoku components to 6h timeframe (already aligned, but need to shift for forward displacement)
    # For signals, we use current Tenkan/Kijun and current Cloud (Senkou A/B already shifted)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Calculate Weekly Pivot from 1d (using prior week's H, L, C)
    # Weekly Pivot: PP = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # R1 = (2 * PP) - Prior Week Low
    # S1 = (2 * PP) - Prior Week High
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly aggregates from daily data
    # We'll use rolling window of 5 days (1 week) to get weekly H, L, C
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly Pivot levels
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = (2 * weekly_pp) - weekly_low
    weekly_s1 = (2 * weekly_pp) - weekly_high
    
    # Align Weekly Pivot to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Volume confirmation: > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(26, 52, 20) + 1  # Kijun needs 26, Senkou B needs 52, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(weekly_pp_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: TK cross bullish, price above cloud, above weekly PP, volume spike
            if (tenkan_aligned[i] > kijun_aligned[i] and  # TK bullish cross
                close[i] > cloud_top and                  # Price above cloud
                close[i] > weekly_pp_aligned[i] and       # Above weekly pivot
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish, price below cloud, below weekly PP, volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and  # TK bearish cross
                  close[i] < cloud_bottom and               # Price below cloud
                  close[i] < weekly_pp_aligned[i] and       # Below weekly pivot
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross bearish OR price falls below cloud OR below weekly PP
            if (tenkan_aligned[i] < kijun_aligned[i] or 
                close[i] < cloud_bottom or 
                close[i] < weekly_pp_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross bullish OR price rises above cloud OR above weekly PP
            if (tenkan_aligned[i] > kijun_aligned[i] or 
                close[i] > cloud_top or 
                close[i] > weekly_pp_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyPivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0