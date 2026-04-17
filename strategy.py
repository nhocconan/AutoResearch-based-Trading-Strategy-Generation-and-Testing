#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d TK Cross confirmation and volume filter.
Long when price breaks above Ichimoku cloud (Senkou Span A/B) AND 1d TK Cross is bullish (Tenkan > Kijun) AND volume > 1.5x 20-period average.
Short when price breaks below Ichimoku cloud AND 1d TK Cross is bearish (Tenkan < Kijun) AND volume > 1.5x 20-period average.
Exit when price re-enters the cloud or TK Cross flips.
Ichimoku provides dynamic support/resistance; TK Cross filters trend alignment; volume confirms breakout strength.
Designed for low trade frequency (12-37/year) to minimize fee drag while capturing strong momentum in both bull and bear markets.
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
    
    # Get 6h data for Ichimoku calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Get 1d data for TK Cross filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    high_52 = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Calculate TK Cross on 1d timeframe
    # Tenkan-sen (1d): (9-period high + 9-period low) / 2
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    # Kijun-sen (1d): (26-period high + 26-period low) / 2
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    # TK Cross bullish: Tenkan > Kijun, bearish: Tenkan < Kijun
    tk_bullish = tenkan_1d > kijun_1d
    tk_bearish = tenkan_1d < kijun_1d
    
    # Calculate volume average (20-period) on 6h
    volume_6h = df_6h['volume'].values
    volume_6h_series = pd.Series(volume_6h)
    volume_ma_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    tk_bullish_aligned = align_htf_to_ltf(prices, df_1d, tk_bullish.astype(float))
    tk_bearish_aligned = align_htf_to_ltf(prices, df_1d, tk_bearish.astype(float))
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tk_bullish_aligned[i]) or np.isnan(tk_bearish_aligned[i]) or
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        tk_bull = tk_bullish_aligned[i] > 0.5
        tk_bear = tk_bearish_aligned[i] > 0.5
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Cloud boundaries: Senkou Span A and B form the cloud
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Long: price breaks above cloud AND bullish TK Cross AND volume > 1.5x avg
            if price > upper_cloud and tk_bull and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud AND bearish TK Cross AND volume > 1.5x avg
            elif price < lower_cloud and tk_bear and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price re-enters cloud OR TK Cross turns bearish
            if price < upper_cloud or not tk_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price re-enters cloud OR TK Cross turns bullish
            if price > lower_cloud or not tk_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TKCross_Volume_Filter"
timeframe = "6h"
leverage = 1.0