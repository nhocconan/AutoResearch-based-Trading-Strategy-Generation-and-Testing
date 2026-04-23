#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d Ichimoku cloud filter with TK cross entry and volume confirmation.
Long when price breaks above Ichimoku cloud (Senkou Span A/B), Tenkan-sen > Kijun-sen (TK cross bullish),
and volume > 1.5x 20-period average. Short when price breaks below cloud, TK cross bearish, and volume spike.
Exit when price re-enters the cloud or opposite TK cross occurs.
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 6h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Ichimoku provides dynamic support/resistance (cloud) and momentum (TK cross) that works in both bull and bear markets.
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
    
    # Calculate 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(df_1d['high']).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(df_1d['low']).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(df_1d['high']).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(df_1d['low']).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(df_1d['high']).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(df_1d['low']).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20)  # Ichimoku needs 52, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # TK cross conditions
        tk_bullish = tenkan_val > kijun_val
        tk_bearish = tenkan_val < kijun_val
        
        if position == 0:
            # Long: Price breaks above cloud AND TK cross bullish AND volume spike
            if (price > upper_cloud and tk_bullish and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud AND TK cross bearish AND volume spike
            elif (price < lower_cloud and tk_bearish and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price re-enters the cloud (between Senkou Span A and B)
            if price > lower_cloud and price < upper_cloud:
                exit_signal = True
            # Secondary exit: Opposite TK cross occurs
            elif position == 1 and tk_bearish:
                exit_signal = True
            elif position == -1 and tk_bullish:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_TK_Cross_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0