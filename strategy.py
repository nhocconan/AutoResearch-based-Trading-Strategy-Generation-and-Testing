#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1w trend filter and volume confirmation.
Long when Tenkan crosses above Kijun AND price > Cloud AND weekly close > weekly open (bullish week) AND volume > 1.5x 20-period average.
Short when Tenkan crosses below Kijun AND price < Cloud AND weekly close < weekly open (bearish week) AND volume > 1.5x 20-period average.
Exit when price crosses back into Cloud or ATR trailing stop (2.5*ATR from extreme).
Ichimoku provides dynamic support/resistance and trend direction. Weekly filter ensures alignment with higher timeframe momentum.
Target trade frequency: 12-37 trades/year per symbol (50-150 total over 4 years) to avoid fee drag.
"""

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
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2.0
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_kijun + lowest_kijun) / 2.0
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan + kijun) / 2.0)
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted 26 periods ahead
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2.0
    
    # The Cloud (Kumo) is between Senkou Span A and Senkou Span B
    # We'll use the current cloud values (not shifted) for price comparison
    # Senkou Span A and B are plotted 26 periods ahead, so to get current cloud we use values from 26 periods ago
    senkou_span_a_lagged = np.roll(senkou_span_a, 26)
    senkou_span_b_lagged = np.roll(senkou_span_b, 26)
    # First 26 values are invalid due to roll
    senkou_span_a_lagged[:26] = np.nan
    senkou_span_b_lagged[:26] = np.nan
    
    # Weekly trend filter: bullish week if weekly close > weekly open
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish week
    weekly_bearish = weekly_close < weekly_open  # True for bearish week
    
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(52, 26, 20, 14)  # Senkou B needs 52, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_span_a_lagged[i]) or np.isnan(senkou_span_b_lagged[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        span_a = senkou_span_a_lagged[i]
        span_b = senkou_span_b_lagged[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bearish_aligned[i] > 0.5
        
        # Cloud boundaries: top is max(span_a, span_b), bottom is min(span_a, span_b)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price > Cloud AND bullish week AND volume spike
            if (tenkan[i-1] <= kijun[i-1] and tenkan_val > kijun_val and  # crossover
                price > cloud_top and 
                weekly_bull and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Tenkan crosses below Kijun AND price < Cloud AND bearish week AND volume spike
            elif (tenkan[i-1] >= kijun[i-1] and tenkan_val < kijun_val and  # crossover
                  price < cloud_bottom and 
                  weekly_bear and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price crosses back into Cloud (between span_a and span_b)
            if position == 1 and price < cloud_top:
                exit_signal = True
            elif position == -1 and price > cloud_bottom:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_1wTrendFilter_VolumeConfirmation_CloudExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0