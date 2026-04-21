#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTK_Cross_v1
Hypothesis: Ichimoku cloud acts as dynamic support/resistance on 6h, while TK cross on 1d provides trend direction. Enter long when price above cloud + TK bullish cross, short when price below cloud + TK bearish cross. Volume confirmation filters false breaks. Designed for low frequency (~15-30 trades/year) to work in both bull/bear regimes by requiring alignment with higher timeframe momentum and institutional volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # === Ichimoku components on 6h (primary timeframe) ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_tenkan + min_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_kijun + min_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    max_senkou = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_senkou = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_senkou + min_senkou) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind (not used for signals)
    
    # === TK Cross on 1d (higher timeframe trend) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen 1d
    max_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    min_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (max_tenkan_1d + min_tenkan_1d) / 2
    
    # Kijun-sen 1d
    max_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    min_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (max_kijun_1d + min_kijun_1d) / 2
    
    # TK Cross: tenkan > kijun = bullish, tenkan < kijun = bearish
    tk_bullish = tenkan_1d > kijun_1d
    tk_bearish = tenkan_1d < kijun_1d
    
    # === Volume confirmation on 6h ===
    volume_6h = prices['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_6h
    
    # Align all HTF indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_6h, 'low': low_6h, 'close': close_6h}), tenkan)
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_6h, 'low': low_6h, 'close': close_6h}), kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_6h, 'low': low_6h, 'close': close_6h}), senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_6h, 'low': low_6h, 'close': close_6h}), senkou_b)
    tk_bullish_aligned = align_htf_to_ltf(prices, df_1d, tk_bullish.astype(float))
    tk_bearish_aligned = align_htf_to_ltf(prices, df_1d, tk_bearish.astype(float))
    vol_ratio_6h_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_6h, 'low': low_6h, 'close': close_6h}), vol_ratio_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tk_bullish_aligned[i]) or np.isnan(tk_bearish_aligned[i]) or
            np.isnan(vol_ratio_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        tk_bull = tk_bullish_aligned[i] > 0.5
        tk_bear = tk_bearish_aligned[i] > 0.5
        vol_ratio = vol_ratio_6h_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Long: price above cloud + TK bullish cross on 1d + volume confirmation
            if price_close > cloud_top and tk_bull and vol_ratio > 1.3:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK bearish cross on 1d + volume confirmation
            elif price_close < cloud_bottom and tk_bear and vol_ratio > 1.3:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses opposite TK signal or closes inside cloud
            if position == 1:
                if tk_bear or (price_close < cloud_top and price_close > cloud_bottom):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if tk_bull or (price_close < cloud_top and price_close > cloud_bottom):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTK_Cross_v1"
timeframe = "6h"
leverage = 1.0