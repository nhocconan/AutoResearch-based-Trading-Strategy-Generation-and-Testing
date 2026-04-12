#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_12h_ichimoku_trend_v1
# Ichimoku system on 12h chart: Tenkan/Kijun cross + price above/below cloud for trend direction.
# Uses Senkou Span A/B from 12h to filter 6s entries. Works in bull markets by buying
# when price > cloud and TK cross bullish, in bear markets by shorting when price < cloud
# and TK cross bearish. Requires volume confirmation to avoid false signals.
# Target: 15-30 trades/year per symbol for low friction.
name = "6h_12h_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Ichimoku calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:  # need enough for Senkou Span B (52 period)
        return np.zeros(n)
    
    # Calculate Ichimoku components on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_12h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_12h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_12h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_12h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_12h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_12h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after warmup for Senkou Span B
        # Skip if Ichimoku not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        # Check volume filter
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price above cloud AND TK cross bullish
        if (close[i] > cloud_top and tenkan_6h[i] > kijun_6h[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short signal: price below cloud AND TK cross bearish
        elif (close[i] < cloud_bottom and tenkan_6h[i] < kijun_6h[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite TK cross or price crosses cloud middle
        elif (tenkan_6h[i] < kijun_6h[i] and position == 1):
            position = 0
            signals[i] = 0.0
        elif (tenkan_6h[i] > kijun_6h[i] and position == -1):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals