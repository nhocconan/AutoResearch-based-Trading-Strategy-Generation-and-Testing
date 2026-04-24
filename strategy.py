#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d ADX regime filter and volume confirmation.
- Uses Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h data.
- Trend filter: 1d ADX > 25 to ensure we only trade in strong trending markets (works in both bull/bear).
- Entry: Long when Tenkan crosses above Kijun AND price is above cloud (Senkou Span A).
         Short when Tenkan crosses below Kijun AND price is below cloud.
- Volume confirmation: volume > 2.0x 20-bar average to avoid false breakouts.
- Designed for 6h timeframe to capture medium-term trends while minimizing whipsaws.
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2.0
    
    # The cloud is between Senkou Span A and Senkou Span B
    # We'll use the current cloud (shifted 26 periods ahead in traditional Ichimoku)
    # But for simplicity, we use the current values as support/resistance
    
    # 1d ADX trend filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up = pd.Series(high_1d).diff()
    down = -(pd.Series(low_1d).diff())
    up = np.where((up > down) & (up > 0), up, 0)
    down = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothed DM
    up_ema = pd.Series(up).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down_ema = pd.Series(down).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * (up_ema / atr)
    minus_di = 100 * (down_ema / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 52)  # Need enough data for Ichimoku (52 periods)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Ichimoku signals
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        price_above_cloud = close[i] > max(senkou_a[i], senkou_b[i])
        price_below_cloud = close[i] < min(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud AND strong trend AND volume
            if tenkan_above_kijun and price_above_cloud and strong_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below cloud AND strong trend AND volume
            elif tenkan_below_kijun and price_below_cloud and strong_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price below cloud
            if tenkan_below_kijun or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price above cloud
            if tenkan_above_kijun or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_ADX25_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0