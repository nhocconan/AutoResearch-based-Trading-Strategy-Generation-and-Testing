#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Ichimoku Cloud with Weekly Trend Filter
# Hypothesis: Ichimoku Cloud provides dynamic support/resistance and trend direction on 6h timeframe.
# In bullish market (weekly price > weekly Kumo), we buy when TK crosses above Kijun and price is above cloud.
# In bearish market (weekly price < weekly Kumo), we sell when TK crosses below Kijun and price is below cloud.
# The weekly timeframe filters out noise and aligns with higher timeframe trend, reducing whipsaws.
# This strategy adapts to both trending and ranging markets by using the cloud as dynamic support/resistance.
# Target: 15-40 trades/year (60-160 over 4 years) to minimize fee drag while capturing significant moves.
name = "6h_ichimoku_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 52:
        return np.zeros(n)
    
    # Ichimoku Cloud on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Weekly trend filter: price vs weekly Kumo (cloud)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Weekly Tenkan-sen and Kijun-sen
    wk_period9_high = pd.Series(weekly_high).rolling(window=9, min_periods=9).max()
    wk_period9_low = pd.Series(weekly_low).rolling(window=9, min_periods=9).min()
    wk_tenkan = (wk_period9_high + wk_period9_low) / 2
    
    wk_period26_high = pd.Series(weekly_high).rolling(window=26, min_periods=26).max()
    wk_period26_low = pd.Series(weekly_low).rolling(window=26, min_periods=26).min()
    wk_kijun = (wk_period26_high + wk_period26_low) / 2
    
    # Weekly Senkou Span A and B
    wk_senkou_a = ((wk_tenkan + wk_kijun) / 2)
    wk_period52_high = pd.Series(weekly_high).rolling(window=52, min_periods=52).max()
    wk_period52_low = pd.Series(weekly_low).rolling(window=52, min_periods=52).min()
    wk_senkou_b = ((wk_period52_high + wk_period52_low) / 2)
    
    # Align weekly Ichimoku components to 6h
    wk_senkou_a_6h = align_htf_to_ltf(prices, df_weekly, wk_senkou_a)
    wk_senkou_b_6h = align_htf_to_ltf(prices, df_weekly, wk_senkou_b)
    
    # Determine weekly Kumo boundaries (cloud)
    wk_kumo_top = np.maximum(wk_senkou_a_6h, wk_senkou_b_6h)
    wk_kumo_bottom = np.minimum(wk_senkou_a_6h, wk_senkou_b_6h)
    
    # Current price relative to weekly cloud
    price_above_weekly_kumo = close > wk_kumo_top
    price_below_weekly_kumo = close < wk_kumo_bottom
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(wk_kumo_top[i]) or np.isnan(wk_kumo_bottom[i])):
            signals[i] = 0.0
            continue
        
        # Current Kumo boundaries (6h cloud)
        kumO_top = np.maximum(senkou_span_a[i], senkou_span_b[i])
        kumO_bottom = np.minimum(senkou_span_a[i], senkou_span_b[i])
        
        if position == 1:  # Long position
            # Exit: TK crosses below Kijun OR price drops below cloud
            if tenkan_sen[i] < kijun_sen[i] or close[i] < kumO_bottom:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: TK crosses above Kijun OR price rises above cloud
            if tenkan_sen[i] > kijun_sen[i] or close[i] > kumO_top:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Bullish setup: weekly uptrend + TK crosses above Kijun + price above cloud
            if (price_above_weekly_kumo[i] and 
                tenkan_sen[i] > kijun_sen[i] and 
                tenkan_sen[i-1] <= kijun_sen[i-1] and  # Cross just happened
                close[i] > kumO_top):
                position = 1
                signals[i] = 0.25
            # Bearish setup: weekly downtrend + TK crosses below Kijun + price below cloud
            elif (price_below_weekly_kumo[i] and 
                  tenkan_sen[i] < kijun_sen[i] and 
                  tenkan_sen[i-1] >= kijun_sen[i-1] and  # Cross just happened
                  close[i] < kumO_bottom):
                position = -1
                signals[i] = -0.25
    
    return signals