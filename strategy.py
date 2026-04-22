#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku Cloud with 1-week trend filter and volume confirmation.
Long when Tenkan-sen crosses above Kijun-sen, price is above Kumo (cloud), and 1-week EMA50 is rising.
Short when Tenkan-sen crosses below Kijun-sen, price is below Kumo, and 1-week EMA50 is falling.
Exit when Tenkan-sen crosses back in opposite direction or price crosses Kumo midpoint.
Ichimoku provides dynamic support/resistance and trend direction; weekly EMA50 filters long-term trend;
volume confirmation reduces false signals. Designed for low trade frequency by requiring multiple confirmations
and using higher timeframe trend filter. Works in both bull and bear markets by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (highest_tenkan + lowest_tenkan) / 2.0
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (highest_kijun + lowest_kijun) / 2.0
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2.0
    
    # Kumo (cloud) top and bottom: Senkou Span A and B
    kumo_top = np.maximum(senkou_span_a, senkou_span_b)
    kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
    kumo_middle = (kumo_top + kumo_bottom) / 2.0  # Midpoint of cloud for exit
    
    # Load 1-week data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1-week EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after enough data for Senkou Span B
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(kumo_top[i]) or 
            np.isnan(kumo_bottom[i]) or np.isnan(kumo_middle[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Ichimoku signals
        tenkan_cross_above = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tenkan_cross_below = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        price_above_kumo = close[i] > kumo_top[i]
        price_below_kumo = close[i] < kumo_bottom[i]
        ema50_rising = ema50_1w_aligned[i] > ema50_1w_aligned[i-1]
        ema50_falling = ema50_1w_aligned[i] < ema50_1w_aligned[i-1]
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above Kumo, weekly EMA50 rising, volume spike
            if (tenkan_cross_above and price_above_kumo and ema50_rising and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, price below Kumo, weekly EMA50 falling, volume spike
            elif (tenkan_cross_below and price_below_kumo and ema50_falling and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Tenkan crosses back in opposite direction OR price crosses Kumo midpoint
            exit_signal = False
            
            if position == 1:
                # Exit long: Tenkan crosses below Kijun OR price crosses below Kumo midpoint
                if tenkan_cross_below or close[i] < kumo_middle[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Tenkan crosses above Kijun OR price crosses above Kumo midpoint
                if tenkan_cross_above or close[i] > kumo_middle[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_1wEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0