#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d/1w trend filter and volume confirmation.
# Long when price is above Kumo (cloud), Tenkan > Kijun, and 1d ADX > 25 (strong trend).
# Short when price is below Kumo, Tenkan < Kijun, and 1d ADX > 25.
# Exit when price crosses back into Kumo or Tenkan/Kijun cross reverses.
# Uses 6h timeframe with 1d ADX and 1w Ichimoku for higher timeframe context.
# Ichimoku performs well in trending markets and avoids whipsaws in ranges.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency.

name = "6h_Ichimoku_1dADX_1wTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for ADX (trend filter)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 30:
        return np.zeros(n)
    
    # Weekly data for Ichimoku (higher timeframe context)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period_kijun)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(period_kijun)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou_span = pd.Series(close).shift(-period_kijun)
    
    # Daily ADX for trend strength
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_d[0] - low_d[0]
    
    # Directional Movement
    plus_dm = np.where((high_d - np.roll(high_d, 1)) > (np.roll(low_d, 1) - low_d), 
                       np.maximum(high_d - np.roll(high_d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_d, 1) - low_d) > (high_d - np.roll(high_d, 1)), 
                        np.maximum(np.roll(low_d, 1) - low_d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_safe
    minus_di = 100 * minus_dm_smooth / atr_safe
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(adx)] = 0
    
    # Align daily ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_d, adx)
    
    # Weekly Ichimoku trend filter: price above/below weekly cloud
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly Tenkan-sen (9-period)
    tenkan_sen_w = (pd.Series(high_w).rolling(window=9, min_periods=9).max() + 
                    pd.Series(low_w).rolling(window=9, min_periods=9).min()) / 2
    
    # Weekly Kijun-sen (26-period)
    kijun_sen_w = (pd.Series(high_w).rolling(window=26, min_periods=26).max() + 
                   pd.Series(low_w).rolling(window=26, min_periods=26).min()) / 2
    
    # Weekly Senkou Span A
    senkou_span_a_w = ((tenkan_sen_w + kijun_sen_w) / 2).shift(26)
    
    # Weekly Senkou Span B (52-period)
    senkou_span_b_w = ((pd.Series(high_w).rolling(window=52, min_periods=52).max() + 
                        pd.Series(low_w).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Determine weekly cloud boundaries (ahead shift already applied)
    wk_span_a = senkou_span_a_w
    wk_span_b = senkou_span_b_w
    
    # Align weekly Ichimoku to 6h timeframe
    tenkan_sen_w_aligned = align_htf_to_ltf(prices, df_w, tenkan_sen_w.values)
    kijun_sen_w_aligned = align_htf_to_ltf(prices, df_w, kijun_sen_w.values)
    wk_span_a_aligned = align_htf_to_ltf(prices, df_w, wk_span_a)
    wk_span_b_aligned = align_htf_to_ltf(prices, df_w, wk_span_b)
    
    # Weekly cloud top and bottom
    wk_kumo_top = np.maximum(wk_span_a_aligned, wk_span_b_aligned)
    wk_kumo_bottom = np.minimum(wk_span_a_aligned, wk_span_b_aligned)
    
    # Weekly trend: price above cloud = uptrend, below cloud = downtrend
    weekly_uptrend = close > wk_kumo_top
    weekly_downtrend = close < wk_kumo_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are valid
    start_idx = max(52, 26, 30)  # Senkou B period, Kijun, ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(wk_kumo_top[i]) or 
            np.isnan(wk_kumo_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 6h Ichimoku signals
        price_above_kumo = close[i] > senkou_span_a[i] and close[i] > senkou_span_b[i]
        price_below_kumo = close[i] < senkou_span_a[i] and close[i] < senkou_span_b[i]
        tenkan_above_kijun = tenkan_sen[i] > kijun_sen[i]
        tenkan_below_kijun = tenkan_sen[i] < kijun_sen[i]
        
        if position == 0:
            # Long: price above Kumo, Tenkan > Kijun, daily ADX > 25, weekly uptrend
            long_cond = (price_above_kumo and tenkan_above_kijun and 
                        adx_aligned[i] > 25 and weekly_uptrend[i])
            # Short: price below Kumo, Tenkan < Kijun, daily ADX > 25, weekly downtrend
            short_cond = (price_below_kumo and tenkan_below_kijun and 
                         adx_aligned[i] > 25 and weekly_downtrend[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below Kumo OR Tenkan < Kijun OR weekly trend turns down
            exit_cond = (not price_above_kumo or not tenkan_above_kijun or 
                        not weekly_uptrend[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Kumo OR Tenkan > Kijun OR weekly trend turns up
            exit_cond = (not price_below_kumo or not tenkan_below_kijun or 
                        not weekly_downtrend[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals