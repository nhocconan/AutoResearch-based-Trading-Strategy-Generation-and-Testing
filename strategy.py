#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with weekly trend filter.
# Tenkan/Kijun cross above/below cloud provides momentum signals.
# Weekly EMA200 determines long-term trend direction: only take longs above weekly EMA200, shorts below.
# This avoids counter-trend trades in strong trends and works in both bull/bear markets by
# following the higher timeframe trend. Target: 15-30 trades/year per symbol.
name = "6h_Ichimoku_TK_Cross_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26  # kijun period
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for long-term trend
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are ready: max(52, 200) + displacement for cloud
    start_idx = max(senkou_span_b_period, 200) + displacement
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate cloud boundaries (shifted forward by displacement)
        senkou_span_a_lead = senkou_span_a[i - displacement] if i >= displacement else senkou_span_a[i]
        senkou_span_b_lead = senkou_span_b[i - displacement] if i >= displacement else senkou_span_b[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_span_a_lead, senkou_span_b_lead)
        cloud_bottom = min(senkou_span_a_lead, senkou_span_b_lead)
        
        price = close[i]
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        weekly_ema = ema_200_1w_aligned[i]
        
        # Determine if price is above or below cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        # TK cross signals
        tk_cross_up = tenkan > kijun
        tk_cross_down = tenkan < kijun
        
        if position == 0:
            # Enter long: TK cross up, price above cloud, and above weekly EMA200
            if tk_cross_up and price_above_cloud and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Enter short: TK cross down, price below cloud, and below weekly EMA200
            elif tk_cross_down and price_below_cloud and price < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross down or price drops below cloud
            if tk_cross_down or price < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross up or price rises above cloud
            if tk_cross_up or price > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals