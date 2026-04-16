#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with TK Cross + Weekly Trend Filter
# Long when price > Ichimoku Cloud AND Tenkan > Kijun (bullish TK cross) AND weekly close > weekly Kumo top
# Short when price < Ichimoku Cloud AND Tenkan < Kijun (bearish TK cross) AND weekly close < weekly Kumo bottom
# Uses Ichimoku components calculated on 6h timeframe with weekly trend alignment
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position size 0.25
# Ichimoku provides dynamic support/resistance and trend identification, weekly filter ensures higher timeframe alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 52 weeks for proper calculation
        return np.zeros(n)
    
    # === Weekly Indicators: Kumo (Cloud) for trend filter ===
    # Calculate Ichimoku components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_1w_tenkan = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_1w_tenkan = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1w = (max_high_1w_tenkan + min_low_1w_tenkan) / 2
    
    # Weekly Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_1w_kijun = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_1w_kijun = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1w = (max_high_1w_kijun + min_low_1w_kijun) / 2
    
    # Weekly Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a_1w = ((tenkan_1w + kijun_1w) / 2)
    
    # Weekly Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_1w_senkou_b = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_1w_senkou_b = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b_1w = ((max_high_1w_senkou_b + min_low_1w_senkou_b) / 2)
    
    # Align weekly Kumo components to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_span_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_1w)
    senkou_span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w)
    
    # === 6h Indicators: Ichimoku Components for entry signals ===
    # 6h Tenkan-sen (Conversion Line): (9-period high + low)/2
    max_high_6h_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_6h_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_6h = (max_high_6h_tenkan + min_low_6h_tenkan) / 2
    
    # 6h Kijun-sen (Base Line): (26-period high + low)/2
    max_high_6h_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_6h_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_6h = (max_high_6h_kijun + min_low_6h_kijun) / 2
    
    # 6h Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a_6h = ((tenkan_6h + kijun_6h) / 2)
    
    # 6h Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    max_high_6h_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_6h_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b_6h = ((max_high_6h_senkou_b + min_low_6h_senkou_b) / 2)
    
    # Current Kumo (Cloud) boundaries: Senkou Span A and B
    # Note: In Ichimoku, the cloud is plotted 26 periods ahead, so we use current values
    # For simplicity in live trading, we use the current Senkou Spans as cloud boundaries
    kumo_top_6h = np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    kumo_bottom_6h = np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(period_senkou_b, 52) + 26  # Enough for all calculations
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(kumo_top_6h[i]) or 
            np.isnan(kumo_bottom_6h[i]) or np.isnan(tenkan_1w_aligned[i]) or 
            np.isnan(kijun_1w_aligned[i]) or np.isnan(senkou_span_a_1w_aligned[i]) or 
            np.isnan(senkou_span_b_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        tenkan_6h_val = tenkan_6h[i]
        kijun_6h_val = kijun_6h[i]
        kumo_top_6h_val = kumo_top_6h[i]
        kumo_bottom_6h_val = kumo_bottom_6h[i]
        tenkan_1w_val = tenkan_1w_aligned[i]
        kijun_1w_val = kijun_1w_aligned[i]
        senkou_span_a_1w_val = senkou_span_a_1w_aligned[i]
        senkou_span_b_1w_val = senkou_span_b_1w_aligned[i]
        
        # Weekly Kumo boundaries for trend filter
        weekly_kumo_top = max(senkou_span_a_1w_val, senkou_span_b_1w_val)
        weekly_kumo_bottom = min(senkou_span_a_1w_val, senkou_span_b_1w_val)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Kumo bottom OR Tenkan crosses below Kijun
            if price < kumo_bottom_6h_val or tenkan_6h_val < kijun_6h_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Kumo top OR Tenkan crosses above Kijun
            if price > kumo_top_6h_val or tenkan_6h_val > kijun_6h_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Bullish conditions: price above cloud, bullish TK cross, weekly trend up
            price_above_cloud = price > kumo_top_6h_val
            bullish_tk = tenkan_6h_val > kijun_6h_val
            weekly_bullish = close_1w[-1] > weekly_kumo_top if len(close_1w) > 0 else False  # Simplified: use last known weekly close
            
            # Bearish conditions: price below cloud, bearish TK cross, weekly trend down
            price_below_cloud = price < kumo_bottom_6h_val
            bearish_tk = tenkan_6h_val < kijun_6h_val
            weekly_bearish = close_1w[-1] < weekly_kumo_bottom if len(close_1w) > 0 else False  # Simplified: use last known weekly close
            
            # LONG: price > cloud AND bullish TK cross AND weekly bullish
            if price_above_cloud and bullish_tk and weekly_bullish:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price < cloud AND bearish TK cross AND weekly bearish
            elif price_below_cloud and bearish_tk and weekly_bearish:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_IchimokuTKCross_WeeklyKumoFilter_V1"
timeframe = "6h"
leverage = 1.0