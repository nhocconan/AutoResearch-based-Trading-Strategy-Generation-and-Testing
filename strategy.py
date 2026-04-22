#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 12h ADX trend filter and volume confirmation.
Long when Tenkan > Kijun + price above cloud (bullish) + 12h ADX > 25 (trending) + volume > 1.5x average.
Short when Tenkan < Kijun + price below cloud (bearish) + 12h ADX > 25 + volume > 1.5x average.
Exit when Tenkan/Kijun cross reverses or ADX < 20 (range) or price re-enters cloud.
Ichimoku provides multi-factor confirmation, ADX filters for trending markets only, volume confirms strength.
Designed for 6h timeframe to balance signal quality and trade frequency (~15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Load 12h data for ADX trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 12h data
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Directional Movement
    dm_plus = df_12h['high'] - df_12h['high'].shift(1)
    dm_minus = df_12h['low'].shift(1) - df_12h['low']
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    atr = tr.rolling(window=14, min_periods=14).mean()
    dm_plus_smooth = dm_plus.rolling(window=14, min_periods=14).mean()
    dm_minus_smooth = dm_minus.rolling(window=14, min_periods=14).mean()
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align 12h ADX to 6s timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Senkou B calculation period
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku signals
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        price_above_cloud = close[i] > max(senkou_a[i], senkou_b[i])
        price_below_cloud = close[i] < min(senkou_a[i], senkou_b[i])
        
        # ADX trend filter
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20  # Exit threshold
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: Bullish Ichimoku + strong trend + volume
            if (tenkan_above_kijun and price_above_cloud and 
                strong_trend and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Ichimoku + strong trend + volume
            elif (tenkan_below_kijun and price_below_cloud and 
                  strong_trend and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Bearish cross OR weak trend OR price re-enters cloud
                if (tenkan_below_kijun or not price_above_cloud or weak_trend):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bullish cross OR weak trend OR price re-enters cloud
                if (tenkan_above_kijun or not price_below_cloud or weak_trend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_12hADX_VolumeFilter"
timeframe = "6h"
leverage = 1.0