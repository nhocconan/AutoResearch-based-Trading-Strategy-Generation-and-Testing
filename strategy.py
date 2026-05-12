#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend
Hypothesis: Ichimoku cloud breakout with daily trend filter on 6h timeframe. 
Enters long when price breaks above Kumo cloud with Tenkan/Kijun cross bullish and daily trend up.
Enters short when price breaks below Kumo cloud with Tenkan/Kijun cross bearish and daily trend down.
Uses daily timeframe for trend filter to avoid counter-trend trades. 
Designed for low trade frequency (12-30/year) to minimize fee drag in 6BTC/ETH markets.
"""

name = "6h_Ichimoku_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou Span A/B, Chikou"""
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
    chikou = np.concatenate([np.full(26, np.nan), close[:-26]]) if len(close) >= 26 else np.full_like(close, np.nan)
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate Ichimoku on 6h data
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Calculate Kumo cloud boundaries (shifted forward by 26 periods for look-ahead free)
    # The actual cloud at time t uses Senkou A/B calculated 26 periods ago
    senkou_a_shifted = np.concatenate([np.full(26, np.nan), senkou_a[:-26]]) if len(senkou_a) >= 26 else np.full_like(senkou_a, np.nan)
    senkou_b_shifted = np.concatenate([np.full(26, np.nan), senkou_b[:-26]]) if len(senkou_b) >= 26 else np.full_like(senkou_b, np.nan)
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    kumo_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Ichimoku warmup
        # Get aligned values for current 6h bar
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        kumo_top_val = kumo_top[i]
        kumo_bottom_val = kumo_bottom[i]
        ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)[i]
        
        # Skip if any required data is NaN
        if (np.isnan(tenkan_val) or np.isnan(kijun_val) or 
            np.isnan(kumo_top_val) or np.isnan(kumo_bottom_val) or 
            np.isnan(ema50_aligned)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Kumo cloud + TK cross bullish + daily uptrend
            if (close[i] > kumo_top_val and 
                tenkan_val > kijun_val and 
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Kumo cloud + TK cross bearish + daily downtrend
            elif (close[i] < kumo_bottom_val and 
                  tenkan_val < kijun_val and 
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Kumo cloud or TK cross turns bearish
            if (close[i] < kumo_top_val or tenkan_val < kijun_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Kumo cloud or TK cross turns bullish
            if (close[i] > kumo_bottom_val or tenkan_val > kijun_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals