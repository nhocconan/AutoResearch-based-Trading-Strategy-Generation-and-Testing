#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume spike.
- Primary timeframe: 6h, HTF: 1d for trend alignment and cloud filter.
- Ichimoku components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52 displacement).
- Entry: Long when price > cloud AND Tenkan > Kijun AND 1d close > 1w EMA50 (bullish regime).
         Short when price < cloud AND Tenkan < Kijun AND 1d close < 1w EMA50 (bearish regime).
- Volume confirmation: current 6h volume > 1.5 * 20-period 6h volume MA.
- Exit: Price crosses back into cloud (Tenkan-Kijun cross or price re-enters cloud).
- Discrete signal size: 0.25 to balance profit potential and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull/bear: 1d/1w regime filter avoids counter-trend trades, Ichimoku cloud acts as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for regime filter (long-term trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1w EMA50 for stronger regime filter (weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Ichimoku calculations on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2.0
    
    # Align HTF indicators to 6h timeframe (completed bars only)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Trend filters: 1d and 1w EMA50 alignment
    bullish_regime = (close > ema_50_1d_aligned) & (close > ema_50_1w_aligned)
    bearish_regime = (close < ema_50_1d_aligned) & (close < ema_50_1w_aligned)
    
    # Cloud top and bottom (Senkou Span A/B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Tenkan/Kijun cross
    tenkan_above_kijun = tenkan > kijun
    tenkan_below_kijun = tenkan < kijun
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 52, 20)  # Need Ichimoku (52), volume MA (20), HTF EMA (50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above cloud AND Tenkan > Kijun AND bullish regime AND volume spike
            if price_above_cloud[i] and tenkan_above_kijun[i] and bullish_regime[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND Tenkan < Kijun AND bearish regime AND volume spike
            elif price_below_cloud[i] and tenkan_below_kijun[i] and bearish_regime[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters cloud OR Tenkan < Kijun (trend weakness)
            if (not price_above_cloud[i]) or (not tenkan_above_kijun[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters cloud OR Tenkan > Kijun (trend weakness)
            if (not price_below_cloud[i]) or (not tenkan_below_kijun[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1d1wEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0