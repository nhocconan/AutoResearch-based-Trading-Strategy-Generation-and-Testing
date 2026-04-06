#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d trend filter and volume confirmation
Hypothesis: Ichimoku provides comprehensive support/resistance and momentum signals.
Using 1d Tenkan/Kijun cross as trend filter and cloud as dynamic S/R reduces whipsaws.
Volume confirmation ensures institutional participation. Works in bull/bear via cloud twist.
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 21-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 21:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[0] = tr[0] if len(tr) > 0 else 0
            for i in range(1, min(len(tr)+1, n)):
                if i-1 < len(tr):
                    atr[i] = (tr[i-1] * 20 + atr[i-1]) / 21 if not np.isnan(atr[i-1]) else tr[i-1]
    
    # Get 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan = np.full(len(high_1d), np.nan)
    for i in range(period_tenkan-1, len(high_1d)):
        tenkan[i] = (np.max(high_1d[i-(period_tenkan-1):i+1]) + np.min(low_1d[i-(period_tenkan-1):i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun = np.full(len(high_1d), np.nan)
    for i in range(period_kijun-1, len(high_1d)):
        kijun[i] = (np.max(high_1d[i-(period_kijun-1):i+1]) + np.min(low_1d[i-(period_kijun-1):i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if i+26 < len(high_1d) and not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i+26] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_b = np.full(len(high_1d), np.nan)
    for i in range(period_senkou_b-1, len(high_1d)):
        senkou_b[i+26] = (np.max(high_1d[i-(period_senkou_b-1):i+1]) + np.min(low_1d[i-(period_senkou_b-1):i+1])) / 2
    
    # Chikou Span (Lagging Span): current close shifted 26 periods back
    chikou = np.full(len(high_1d), np.nan)
    for i in range(26, len(high_1d)):
        chikou[i] = close_1d[i-26]
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_6h = align_htf_to_ltf(prices, df_1d, chikou)
    
    # Get 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_1d_6h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 60  # Need enough data for Ichimoku calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(chikou_6h[i]) or np.isnan(vol_ma_1d_6h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.3x daily average volume (scaled)
        vol_threshold = vol_ma_1d_6h[i] / 4.0 * 1.3  # 4x 6h bars in 1d
        volume_filter = volume[i] > vol_threshold
        
        # Cloud calculations
        cloud_top = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        # Trend filter: Tenkan/Kijun cross
        bullish_cross = tenkan_6h[i] > kijun_6h[i]
        bearish_cross = tenkan_6h[i] < kijun_6h[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        price_in_cloud = close[i] >= cloud_bottom and close[i] <= cloud_top
        
        # Chikou confirmation: Chikou vs price 26 periods ago
        chikou_confirmed = not np.isnan(chikou_6h[i]) and chikou_6h[i] > close[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below cloud OR Tenkan/Kijun cross turns bearish
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] < cloud_bottom or
                bearish_cross or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above cloud OR Tenkan/Kijun cross turns bullish
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] > cloud_top or
                bullish_cross or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries with minimum holding period
            if bars_since_entry >= 24:  # At least 4 bars (24h) between entries
                # Long: price above cloud + bullish TK cross + Chikou confirmation + volume
                long_condition = (price_above_cloud and 
                                bullish_cross and 
                                chikou_confirmed and 
                                volume_filter)
                
                # Short: price below cloud + bearish TK cross + Chikou confirmation + volume
                short_condition = (price_below_cloud and 
                                 bearish_cross and 
                                 not chikou_confirmed and  # Chikou below price for bearish
                                 volume_filter)
                
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals