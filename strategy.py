#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with weekly bias and volume confirmation.
# Long when price is above Kumo (cloud), Tenkan > Kijun, and weekly trend is up (price > weekly Kumo).
# Short when price is below Kumo, Tenkan < Kijun, and weekly trend is down (price < weekly Kumo).
# Exit when Tenkan crosses Kijun in opposite direction.
# Uses Ichimoku on 6h for entry timing and weekly Ichimoku for trend filter.
# Ichimoku provides multiple confirmation layers (Tenkan/Kijun cross, Kumo twist, price vs cloud).
# Weekly filter ensures we trade with the higher timeframe trend, reducing counter-trend whipsaws.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
# Works in bull markets via trend following and in bear markets via short signals from weekly downtrend.

name = "6h_Ichimoku_Cloud_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted 26 periods ahead
    # For cloud at time t, we use values from t-26
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Weekly Ichimoku for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku components
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    wclose = df_1w['close'].values
    
    # Weekly Tenkan-sen (9-period)
    w_period9_high = pd.Series(whigh).rolling(window=9, min_periods=9).max().values
    w_period9_low = pd.Series(wlow).rolling(window=9, min_periods=9).min().values
    w_tenkan = (w_period9_high + w_period9_low) / 2
    
    # Weekly Kijun-sen (26-period)
    w_period26_high = pd.Series(whigh).rolling(window=26, min_periods=26).max().values
    w_period26_low = pd.Series(wlow).rolling(window=26, min_periods=26).min().values
    w_kijun = (w_period26_high + w_period26_low) / 2
    
    # Weekly Senkou Span A
    w_senkou_a = (w_tenkan + w_kijun) / 2
    
    # Weekly Senkou Span B (52-period)
    w_period52_high = pd.Series(whigh).rolling(window=52, min_periods=52).max().values
    w_period52_low = pd.Series(wlow).rolling(window=52, min_periods=52).min().values
    w_senkou_b = (w_period52_high + w_period52_low) / 2
    
    # Weekly Kumo (Cloud)
    w_senkou_a_shifted = np.roll(w_senkou_a, 26)
    w_senkou_b_shifted = np.roll(w_senkou_b, 26)
    w_senkou_a_shifted[:26] = np.nan
    w_senkou_b_shifted[:26] = np.nan
    
    # Align weekly Ichimoku components to 6h timeframe
    w_tenkan_aligned = align_htf_to_ltf(prices, df_1w, w_tenkan)
    w_kijun_aligned = align_htf_to_ltf(prices, df_1w, w_kijun)
    w_senkou_a_aligned = align_htf_to_ltf(prices, df_1w, w_senkou_a_shifted)
    w_senkou_b_aligned = align_htf_to_ltf(prices, df_1w, w_senkou_b_shifted)
    
    # Weekly trend: price above/below weekly Kumo
    # Kumo top = max(Senkou A, Senkou B), Kumo bottom = min(Senkou A, Senkou B)
    w_kumo_top = np.maximum(w_senkou_a_aligned, w_senkou_b_aligned)
    w_kumo_bottom = np.minimum(w_senkou_a_aligned, w_senkou_b_aligned)
    weekly_uptrend = wclose > w_kumo_top  # Using current weekly close for trend
    weekly_downtrend = wclose < w_kumo_bottom
    
    # Align weekly trend to 6h
    # We need to align the weekly close series first
    wclose_series = pd.Series(wclose)
    wclose_aligned = align_htf_to_ltf(prices, df_1w, wclose_series.values)
    weekly_uptrend_aligned = wclose_aligned > w_kumo_top
    weekly_downtrend_aligned = wclose_aligned < w_kumo_bottom
    
    # 6h price vs Kumo
    kumo_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    kumo_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    price_above_kumo = close > kumo_top
    price_below_kumo = close < kumo_bottom
    
    # Tenkan/Kijun cross
    tenkan_above_kijun = tenkan > kijun
    tenkan_below_kijun = tenkan < kijun
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26, 20)  # Sufficient warmup for Ichimoku and volume
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(kumo_top[i]) or 
            np.isnan(kumo_bottom[i]) or np.isnan(w_tenkan_aligned[i]) or 
            np.isnan(w_kijun_aligned[i]) or np.isnan(w_kumo_top[i]) or 
            np.isnan(w_kumo_bottom[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above Kumo, Tenkan > Kijun, weekly uptrend, volume filter
            long_cond = (price_above_kumo[i] and tenkan_above_kijun[i] and 
                        weekly_uptrend_aligned[i] and volume_filter[i])
            # Short conditions: price below Kumo, Tenkan < Kijun, weekly downtrend, volume filter
            short_cond = (price_below_kumo[i] and tenkan_below_kijun[i] and 
                         weekly_downtrend_aligned[i] and volume_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun (or price drops below Kumo)
            if tenkan_below_kijun[i] or price_below_kumo[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun (or price rises above Kumo)
            if tenkan_above_kijun[i] or price_above_kumo[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals