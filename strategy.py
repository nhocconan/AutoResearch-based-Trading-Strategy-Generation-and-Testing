#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud system with 1d ADX trend filter and volume confirmation.
Long when price above Kumo (cloud), Tenkan > Kijun, and strong ADX trend with volume.
Short when price below Kumo, Tenkan < Kijun, and strong ADX trend with volume.
Exit when price crosses Kijun or ADX weakens.
Ichimoku provides dynamic support/resistance; ADX filters ranging markets.
Designed for low trade frequency (15-35/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Ichimoku and ADX - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily
    high_d = pd.Series(df_daily['high'].values)
    low_d = pd.Series(df_daily['low'].values)
    close_d = pd.Series(df_daily['close'].values)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = high_d.rolling(window=9, min_periods=9).max()
    period9_low = low_d.rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = high_d.rolling(window=26, min_periods=26).max()
    period26_low = low_d.rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = high_d.rolling(window=52, min_periods=52).max()
    period52_low = low_d.rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2).shift(26)
    
    # Kumo (Cloud): Senkou Span A and B
    # For cloud color: Senkou A > Senkou B = bullish cloud, else bearish
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = high_d - low_d
    tr2 = (high_d - close_d.shift(1)).abs()
    tr3 = (low_d - close_d.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_d.diff()
    down_move = -low_d.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_d)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_d)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_d = dx.rolling(window=14, min_periods=14).mean()
    
    # Align Ichimoku and ADX to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_daily, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_daily, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_daily, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_daily, senkou_b.values)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx_d.values)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Need enough data for Ichimoku
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Kumo (cloud) boundaries
        upper_kumo = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_kumo = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: Price above cloud, Tenkan > Kijun, strong ADX, volume spike
            if (close[i] > upper_kumo and 
                tenkan_aligned[i] > kijun_aligned[i] and 
                adx_aligned[i] > 25 and  # Strong trend
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, Tenkan < Kijun, strong ADX, volume spike
            elif (close[i] < lower_kumo and 
                  tenkan_aligned[i] < kijun_aligned[i] and 
                  adx_aligned[i] > 25 and  # Strong trend
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Kijun OR ADX weakens
                if close[i] < kijun_aligned[i] or adx_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Kijun OR ADX weakens
                if close[i] > kijun_aligned[i] or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_1dADX_Volume"
timeframe = "6h"
leverage = 1.0
#%%