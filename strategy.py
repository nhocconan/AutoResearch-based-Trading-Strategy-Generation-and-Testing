#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d ADX filter for trend confirmation.
Long when price is above cloud, Tenkan > Kijun, and 1d ADX > 25 (strong trend).
Short when price is below cloud, Tenkan < Kijun, and 1d ADX > 25.
Exit when price crosses cloud or Tenkan/Kijun cross reverses.
Ichimoku provides dynamic support/resistance; ADX filters weak trends.
Designed for low trade frequency (10-25/year) to minimize fee drag.
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
    
    # Load daily data for ADX filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Ichimoku Cloud (9, 26, 52) on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Calculate 1d ADX (14-period)
    high_d = pd.Series(df_daily['high'].values)
    low_d = pd.Series(df_daily['low'].values)
    close_d = pd.Series(df_daily['close'].values)
    
    # True Range
    tr1 = high_d - low_d
    tr2 = abs(high_d - close_d.shift(1))
    tr3 = abs(low_d - close_d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_d.diff()
    down_move = -low_d.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    plus_di_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean()
    minus_di_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_di_smooth / atr_d)
    minus_di = 100 * (minus_di_smooth / atr_d)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_d = dx.rolling(window=14, min_periods=14).mean()
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx_d.values)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Wait for Senkou B calculation
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(adx_aligned[i])):
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
        
        # Cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long: Price above cloud, Tenkan > Kijun, strong ADX
            if (close[i] > cloud_top and 
                tenkan[i] > kijun[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, Tenkan < Kijun, strong ADX
            elif (close[i] < cloud_bottom and 
                  tenkan[i] < kijun[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price below cloud OR Tenkan < Kijun
                if close[i] < cloud_top or tenkan[i] < kijun[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above cloud OR Tenkan > Kijun
                if close[i] > cloud_bottom or tenkan[i] > kijun[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_IchimokuCloud_1dADX"
timeframe = "6h"
leverage = 1.0
#%%