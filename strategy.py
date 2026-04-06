#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d ADX filter and volume confirmation
# Enter long when: Tenkan > Kijun (bullish TK cross), price > Kumo cloud (bullish), ADX > 25 (trending), volume > 1.5x average
# Enter short when: Tenkan < Kijun (bearish TK cross), price < Kumo cloud (bearish), ADX > 25, volume > 1.5x average
# Exit when: TK cross reverses OR price crosses Kumo in opposite direction
# Uses Ichimoku for trend/momentum, ADX to avoid ranging markets, volume for confirmation
# Ichimoku works well in 6h timeframe as it captures medium-term trends
# ADX filter prevents whipsaws in sideways markets (critical for 2022-2025 bear/ranging conditions)
# Target: 50-150 trades over 4 years

name = "6h_ichimoku_1dadx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # Need enough data for Ichimoku (52 periods)
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Ichimoku Cloud components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (14-period)
    def smooth_series(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values use Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = smooth_series(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku is fully calculated
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine Kumo (cloud) boundaries
        # Senkou Span A and B are plotted 26 periods ahead
        # For current price, we need values from 26 periods ago
        idx_a = i - 26
        idx_b = i - 26
        if idx_a >= 0 and idx_b >= 0:
            senkou_a_current = senkou_a[idx_a]
            senkou_b_current = senkou_b[idx_b]
            
            # Kumo top and bottom
            kumo_top = max(senkou_a_current, senkou_b_current)
            kumo_bottom = min(senkou_a_current, senkou_b_current)
            
            # Kumo color: green if Senkou A > Senkou B (bullish), red otherwise
            kumo_bullish = senkou_a_current > senkou_b_current
        else:
            # Not enough data for cloud projection
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: TK cross turns bearish OR price drops below Kumo bottom
            if tenkan[i] < kijun[i] or close[i] < kumo_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross turns bullish OR price rises above Kumo top
            if tenkan[i] > kijun[i] or close[i] > kumo_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross + price vs Kumo + ADX trend + volume
            if (adx_aligned[i] > 25 and  # Strong trend
                volume[i] > volume_threshold[i]):  # Volume confirmation
                
                # Bullish: TK cross bullish, price above Kumo (bullish cloud)
                if tenkan[i] > kijun[i] and close[i] > kumo_top and kumo_bullish:
                    signals[i] = 0.25
                    position = 1
                # Bearish: TK cross bearish, price below Kumo (bearish cloud)
                elif tenkan[i] < kijun[i] and close[i] < kumo_bottom and not kumo_bullish:
                    signals[i] = -0.25
                    position = -1
    
    return signals