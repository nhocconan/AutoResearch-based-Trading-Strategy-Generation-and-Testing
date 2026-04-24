#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d ADX trend filter and volume spike confirmation.
- Uses Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h candles.
- Long: Tenkan > Kijun AND price above cloud (Senkou Span A/B) AND 1d ADX > 25 AND volume > 2.0x 20-bar average.
- Short: Tenkan < Kijun AND price below cloud AND 1d ADX > 25 AND volume > 2.0x 20-bar average.
- Ichimoku cloud acts as dynamic support/resistance; ADX > 25 ensures trending market (avoids chop).
- Volume confirmation filters breakouts with strong participation.
- Designed for 6h timeframe to capture medium-term trends with controlled frequency.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Ichimoku components (6h timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_9 + lowest_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_26 + lowest_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    period_senkou_b = 52
    highest_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((highest_52 + lowest_52) / 2.0)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind (not used for entry)
    
    # Align Senkou Span A/B to current time (they are already shifted in calculation)
    # Senkou A/B are plotted 26 periods ahead, so to get current cloud, we use values shifted back 26
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # 1d ADX trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0.0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0.0)
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    period_adx = 14
    alpha = 1.0 / period_adx
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initialize first value
    if len(tr) > 0:
        atr[0] = tr[0]
        dm_plus_smooth[0] = dm_plus[0]
        dm_minus_smooth[0] = dm_minus[0]
    
    # Wilder's smoothing
    for i in range(1, len(tr)):
        atr[i] = atr[i-1] + alpha * (tr[i] - atr[i-1])
        dm_plus_smooth[i] = dm_plus_smooth[i-1] + alpha * (dm_plus[i] - dm_plus_smooth[i-1])
        dm_minus_smooth[i] = dm_minus_smooth[i-1] + alpha * (dm_minus[i] - dm_minus_smooth[i-1])
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0.0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0.0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0.0)
    adx = np.zeros_like(dx)
    if len(dx) > 0:
        adx[0] = dx[0]
        for i in range(1, len(dx)):
            adx[i] = adx[i-1] + alpha * (dx[i] - adx[i-1])
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 52 + 26, 34)  # enough for Ichimoku and ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_lagged[i]) or np.isnan(senkou_b_lagged[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_a_lagged[i], senkou_b_lagged[i])
        lower_cloud = min(senkou_a_lagged[i], senkou_b_lagged[i])
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # ADX trend filter (> 25 = trending market)
        strong_trend = adx_aligned[i] > 25.0
        
        if position == 0:
            # Long: Tenkan > Kijun AND price above cloud AND strong trend AND volume confirmation
            if (tenkan[i] > kijun[i] and 
                close[i] > upper_cloud and 
                strong_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun AND price below cloud AND strong trend AND volume confirmation
            elif (tenkan[i] < kijun[i] and 
                  close[i] < lower_cloud and 
                  strong_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan < Kijun OR price below cloud OR loss of trend
            if (tenkan[i] < kijun[i] or 
                close[i] < lower_cloud or 
                adx_aligned[i] < 20.0):  # exit if ADX drops below 20 (trend weakening)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan > Kijun OR price above cloud OR loss of trend
            if (tenkan[i] > kijun[i] or 
                close[i] > upper_cloud or 
                adx_aligned[i] < 20.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_ADX25_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0