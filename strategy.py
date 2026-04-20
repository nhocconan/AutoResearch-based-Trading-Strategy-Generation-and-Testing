#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Ichimoku Cloud with 1-week trend filter + volume confirmation
# Uses Kumo (cloud) breakout/trend following: price above/below cloud + TK cross
# Weekly trend filter: price above/below weekly Kumo to determine bias
# Volume confirmation: require volume > 1.3x 20-period average
# Designed to work in both bull and bear markets by adapting to trend direction
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components on weekly timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a_1w = ((tenkan_sen_1w + kijun_sen_1w) / 2).values
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b_1w = ((pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2).values
    
    # Align weekly Ichimoku to 12h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen_1w.values)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen_1w.values)
    span_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_1w)
    span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w)
    
    # Load daily data for Ichimoku (main signal)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on daily timeframe
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2).values
    senkou_span_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).values
    
    # Align daily Ichimoku to 12h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d.values)
    span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Load 12h data for price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if NaN in indicators
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(span_a_1w_aligned[i]) or np.isnan(span_b_1w_aligned[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(span_a_1d_aligned[i]) or np.isnan(span_b_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend bias (price vs weekly cloud)
        weekly_kumo_top = max(span_a_1w_aligned[i], span_b_1w_aligned[i])
        weekly_kumo_bottom = min(span_a_1w_aligned[i], span_b_1w_aligned[i])
        price_above_weekly_kumo = close[i] > weekly_kumo_top
        price_below_weekly_kumo = close[i] < weekly_kumo_bottom
        
        # Determine daily Ichimoku signals
        daily_kumo_top = max(span_a_1d_aligned[i], span_b_1d_aligned[i])
        daily_kumo_bottom = min(span_a_1d_aligned[i], span_b_1d_aligned[i])
        
        # TK cross signals
        tk_cross_bull = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_cross_bear = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Price vs cloud
        price_above_daily_kumo = close[i] > daily_kumo_top
        price_below_daily_kumo = close[i] < daily_kumo_bottom
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Enter long: bullish TK cross + price above daily cloud + weekly bullish bias + volume
            long_signal = (tk_cross_bull and price_above_daily_kumo and 
                          price_above_weekly_kumo and has_volume)
            
            # Enter short: bearish TK cross + price below daily cloud + weekly bearish bias + volume
            short_signal = (tk_cross_bear and price_below_daily_kumo and 
                           price_below_weekly_kumo and has_volume)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish TK cross or price below daily cloud
            exit_signal = (tk_cross_bear or price_below_daily_kumo)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish TK cross or price above daily cloud
            exit_signal = (tk_cross_bull or price_above_daily_kumo)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Ichimoku_WeeklyTrendFilter_Volume"
timeframe = "12h"
leverage = 1.0