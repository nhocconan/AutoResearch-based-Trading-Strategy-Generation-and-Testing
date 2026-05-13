#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend
Hypothesis: Ichimoku cloud (TK cross + price above/below cloud) with 1d trend filter and volume confirmation works in both bull and bear markets.
Long when TK cross bullish, price above cloud, 1d uptrend, volume spike. Short when TK cross bearish, price below cloud, 1d downtrend, volume spike.
Exit on opposite TK cross or trend reversal. Uses 12h trend for additional confirmation.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    # But for cloud calculation, we need current values
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Cloud boundaries (use current Senkou spans)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # TK Cross signals
    tk_bullish = tenkan > kijun
    tk_bearish = tenkan < kijun
    
    # Price relative to cloud
    price_above_cloud = close > upper_cloud
    price_below_cloud = close < lower_cloud
    
    # 12h trend filter (MTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 26:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Tenkan and Kijun for trend
    high_9_12h = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    low_9_12h = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan_12h = (high_9_12h + low_9_12h) / 2
    
    high_26_12h = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    low_26_12h = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun_12h = (high_26_12h + low_26_12h) / 2
    
    tk_bullish_12h = tenkan_12h > kijun_12h
    tk_bearish_12h = tenkan_12h < kijun_12h
    
    tk_bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, tk_bullish_12h)
    tk_bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, tk_bearish_12h)
    
    # 1d trend filter (additional HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_26_1d = pd.Series(close_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(close_1d).rolling(window=26, min_periods=26).min().values
    # Using price vs 26-period high/low as trend proxy
    uptrend_1d = close_1d > (high_26_1d + low_26_1d) / 2
    downtrend_1d = close_1d < (high_26_1d + low_26_1d) / 2
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after 52 for full Ichimoku
        # Get values
        tk_bull = tk_bullish[i]
        tk_bear = tk_bearish[i]
        price_above = price_above_cloud[i]
        price_below = price_below_cloud[i]
        vol_conf = volume_conf[i]
        tk_bull_12h = tk_bullish_12h_aligned[i]
        tk_bear_12h = tk_bearish_12h_aligned[i]
        uptrend_1d = uptrend_1d_aligned[i]
        downtrend_1d = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: TK bullish, price above cloud, 12h bullish, 1d uptrend, volume confirmation
            if tk_bull and price_above and tk_bull_12h and uptrend_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: TK bearish, price below cloud, 12h bearish, 1d downtrend, volume confirmation
            elif tk_bear and price_below and tk_bear_12h and downtrend_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK bearish or price below cloud or 1d trend turns down
            if tk_bear or not price_above or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK bullish or price above cloud or 1d trend turns up
            if tk_bull or not price_below or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals