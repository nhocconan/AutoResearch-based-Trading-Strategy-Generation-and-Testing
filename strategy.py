#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend
Hypothesis: Use Ichimoku cloud from daily timeframe for trend filter and TK cross on 6h for entries.
In bull markets: price above cloud + TK cross bullish = long.
In bear markets: price below cloud + TK cross bearish = short.
Cloud acts as dynamic support/resistance, reducing whipsaws in ranging markets.
Targets 15-30 trades/year by requiring TK cross + cloud alignment.
Works in both bull and bear via cloud filter and directional bias.
"""

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, senkouA, senkouB, chikou"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = np.roll(close, 26)  # Will be handled by alignment
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Ichimoku cloud (Senkou A/B) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)

    # Calculate Ichimoku on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # The cloud is between Senkou A and Senkou B
    # We need to align these to 6h timeframe
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Get 6h data for TK cross (Tenkan/Kijun) - calculate on 6h directly
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high + period9_low) / 2
    
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high + period26_low) / 2
    
    # TK cross signals: Tenkan crossing above/below Kijun
    tk_cross_bull = (tenkan_6h > kijun_6h) & (tenkan_6h <= kijun_6h)  # Crossed above
    tk_cross_bear = (tenkan_6h < kijun_6h) & (tenkan_6h >= kijun_6h)  # Crossed below
    
    # Volume confirmation: current volume > 1.3x average of last 24 periods (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after warmup for Ichimoku
        if (np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud boundaries (Senkou A and B)
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Price position relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        price_in_cloud = (close[i] >= cloud_bottom) & (close[i] <= cloud_top)

        if position == 0:
            # LONG: price above cloud + TK cross bullish + volume
            if price_above_cloud and tk_cross_bull[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price below cloud + TK cross bearish + volume
            elif price_below_cloud and tk_cross_bear[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below cloud OR TK cross bearish
            if price_below_cloud or tk_cross_bear[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above cloud OR TK cross bullish
            if price_above_cloud or tk_cross_bull[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals