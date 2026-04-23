#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d ADX filter and volume confirmation.
Long when Tenkan-sen crosses above Kijun-sen, price is above cloud, 1d ADX > 25, and volume > 1.5x average.
Short when Tenkan-sen crosses below Kijun-sen, price is below cloud, 1d ADX > 25, and volume > 1.5x average.
Uses 6h timeframe targeting 50-150 total trades over 4 years. Ichimoku provides trend, support/resistance, and momentum.
ADX filter ensures we only trade in trending markets, reducing whipsaw in ranging conditions. Volume confirmation adds conviction.
Works in both bull and bear markets by aligning with higher timeframe trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ichimoku_cloud(high, low, close, tenkan=9, kijun=26, senkou_b=52):
    """Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=tenkan).max()
    period9_low = pd.Series(low).rolling(window=tenkan).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=kijun).max()
    period26_low = pd.Series(low).rolling(window=kijun).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=senkou_b).max()
    period52_low = pd.Series(low).rolling(window=senkou_b).min()
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    # Not used in signals to avoid look-ahead
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values

def adx(high, low, close, period=14):
    """Average Directional Index"""
    # True Range
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = abs(pd.Series(high) - pd.Series(close).shift())
    tr3 = abs(pd.Series(low) - pd.Series(close).shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(high) - pd.Series(high).shift()
    dm_minus = pd.Series(low).shift() - pd.Series(low)
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=1/period, adjust=False).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/period, adjust=False).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX
    adx_1d = adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate Ichimoku on 6h timeframe
    tenkan_sen, kijun_sen, senkou_a, senkou_b = ichimoku_cloud(high, low, close)
    
    # Align HTF ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        tenkan_val = tenkan_sen[i]
        kijun_val = kijun_sen[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        adx_val = adx_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, ADX > 25, volume confirmation
            if (tenkan_val > kijun_val and  # Tenkan/Kijun cross (current bar)
                price > upper_cloud and     # Price above cloud
                adx_val > 25 and            # Strong trend
                vol_current > 1.5 * vol_ma_val):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, price below cloud, ADX > 25, volume confirmation
            elif (tenkan_val < kijun_val and  # Tenkan/Kijun cross (current bar)
                  price < lower_cloud and     # Price below cloud
                  adx_val > 25 and            # Strong trend
                  vol_current > 1.5 * vol_ma_val):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Tenkan crosses below Kijun OR price falls below cloud
                if tenkan_val < kijun_val or price < lower_cloud:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Tenkan crosses above Kijun OR price rises above cloud
                if tenkan_val > kijun_val or price > upper_cloud:
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