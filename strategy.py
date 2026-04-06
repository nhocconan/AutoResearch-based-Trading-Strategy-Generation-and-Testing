#!/usr/bin/env python3
"""
6h Ichimoku Kumo Breakout with Volume and Trend Filter
Hypothesis: Ichimoku system on 1d timeframe provides robust trend direction and support/resistance.
60-period Tenkan/Kijun cross combined with Kumo (cloud) filter from daily timeframe,
entered on 6h timeframe with volume confirmation. Works in bull (buy above cloud) and
bear (sell below cloud) via Kumo color filter. Low frequency due to multi-condition
requirements.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_kumo_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A/B, Chikou"""
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
    # Not used for signals but calculated for completeness
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Ichimoku (once before loop)
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku on daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high_daily, low_daily, close_daily)
    
    # Kumo (cloud) color: green if Senkou A > Senkou B (bullish), red otherwise
    kumo_green = senkou_a > senkou_b
    
    # TK cross signals: bullish when Tenkan > Kijun
    tk_bullish = tenkan > kijun
    
    # Align all indicators to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_daily, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_daily, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_daily, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_daily, senkou_b)
    kumo_green_6h = align_htf_to_ltf(prices, df_daily, kumo_green)
    tk_bullish_6h = align_htf_to_ltf(prices, df_daily, tk_bullish)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (need 52 periods for Senkou B)
    start = 60
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(kumo_green_6h[i]) or np.isnan(tk_bullish_6h[i]) or
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Cloud boundaries: top is senkou_a, bottom is senkou_b
        kumo_top = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        kumo_bottom = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        # Check exits: price crosses Kumo in opposite direction or TK cross reverses
        if position == 1:  # long position
            # Exit: price falls below cloud OR TK cross turns bearish
            if (close[i] <= kumo_bottom or not tk_bullish_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above cloud OR TK cross turns bullish
            if (close[i] >= kumo_top or tk_bullish_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price breaks Kumo with TK alignment and volume
            # Bullish: price breaks above cloud AND TK bullish AND volume confirmation
            bull_breakout = close[i] > kumo_top
            bull_entry = bull_breakout and tk_bullish_6h[i] and volume[i] > vol_ema[i] * 1.5
            
            # Bearish: price breaks below cloud AND TK bearish AND volume confirmation
            bear_breakout = close[i] < kumo_bottom
            bear_entry = bear_breakout and (not tk_bullish_6h[i]) and volume[i] > vol_ema[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals