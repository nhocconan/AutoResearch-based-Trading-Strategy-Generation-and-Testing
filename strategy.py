#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_V1
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter (EMA50). Uses Tenkan/Kijun cross above/below cloud + volume confirmation.
Works in bull via cloud breakouts above Kumo, in bear via breakdowns below Kumo. Position size 0.25 balances risk.
Target: ~12-37 trades/year per symbol (50-150 total over 4 years). Uses 6h primary with 1d HTF for trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Ichimoku Components ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # need 26*2 for Senkou B
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2.0)
    
    # Align Ichimoku components to 6h timeframe (they are already calculated on 6h)
    # For cloud, we need current Senkou A/B (no alignment needed as same TF)
    # But for Tenkan/Kijun cross, we use current values
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after Senkou B calculation is ready
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) 
            or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Cloud boundaries: Senkou A and B form the cloud
        top_cloud = max(senkou_a[i], senkou_b[i])
        bottom_cloud = min(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud + volume + 1d uptrend
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]  # bullish TK cross
                and price > top_cloud  # price above cloud
                and vol_ok
                and price > ema_50_1d_aligned[i]):  # 1d uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below cloud + volume + 1d downtrend
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]  # bearish TK cross
                  and price < bottom_cloud  # price below cloud
                  and vol_ok
                  and price < ema_50_1d_aligned[i]):  # 1d downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below cloud OR Tenkan crosses below Kijun
            if price < bottom_cloud or (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above cloud OR Tenkan crosses above Kijun
            if price > top_cloud or (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_V1"
timeframe = "6h"
leverage = 1.0