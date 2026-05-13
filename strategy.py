#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_Trend
Hypothesis: Ichimoku cloud breakout with daily trend filter and volume confirmation works across market regimes.
- Long when price breaks above Kumo (cloud), Tenkan > Kijun, and price > daily EMA50
- Short when price breaks below Kumo, Tenkan < Kijun, and price < daily EMA50
- Volume confirmation filters false breakouts
- Exit when price re-enters Kumo or trend reverses
Target: 25-40 trades/year per symbol (100-160 total over 4 years)
Works in bull (trend continuation) and bear (mean reversion from oversold/overbought) markets
"""

name = "6h_Ichimoku_Cloud_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

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
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): current close plotted 26 periods behind
    # Not used for entry/exit to avoid look-ahead
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For cloud breakout, we compare current price to the cloud that was formed 26 periods ago
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid due to shift
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Daily trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku calculations are valid
        # Skip if any values are NaN
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]):
            signals[i] = 0.0
            continue
            
        # Get values
        price = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price above cloud, Tenkan > Kijun (bullish), daily uptrend, volume confirmation
            if (price > cloud_top_val and 
                tenkan_val > kijun_val and 
                uptrend_htf and 
                vol_conf):
                signals[i] = 0.25
                position = 1
            # SHORT: price below cloud, Tenkan < Kijun (bearish), daily downtrend, volume confirmation
            elif (price < cloud_bottom_val and 
                  tenkan_val < kijun_val and 
                  downtrend_htf and 
                  vol_conf):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price re-enters cloud or trend turns bearish
            if (price < cloud_top_val or 
                not uptrend_htf):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price re-enters cloud or trend turns bullish
            if (price > cloud_bottom_val or 
                not downtrend_htf):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals