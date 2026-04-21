#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Uses Ichimoku cloud on 6h for trend and breakout signals, filtered by 1d EMA50 trend and volume spike confirmation. Designed for low trade frequency (~15-30/year) to minimize fee drag. Works in bull markets via cloud breakouts and in bear markets via cloud breakdowns with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === Ichimoku components on 6h (primary timeframe) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low) / 2, shifted 26 periods ahead
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((highest_senkou_b + lowest_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used in signals to avoid look-ahead
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_1d = ema_50_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long: price breaks above cloud + Tenkan > Kijun (bullish) + 1d uptrend + volume spike
            if (price_close > cloud_top and tenkan[i] > kijun[i] and 
                price_close > trend_1d and vol_spike > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud + Tenkan < Kijun (bearish) + 1d downtrend + volume spike
            elif (price_close < cloud_bottom and tenkan[i] < kijun[i] and 
                  price_close < trend_1d and vol_spike > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price re-enters the cloud or Tenkan/Kijun cross reverses
            if position == 1:
                if price_close < cloud_top or tenkan[i] < kijun[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > cloud_bottom or tenkan[i] > kijun[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0