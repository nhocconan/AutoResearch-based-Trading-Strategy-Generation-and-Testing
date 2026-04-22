#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku Cloud with weekly trend filter.
Long when price > Kumo and Tenkan > Kijun, with weekly trend bullish.
Short when price < Kumo and Tenkan < Kijun, with weekly trend bearish.
Ichimoku provides multi-line support/resistance and momentum signals.
Weekly trend filter ensures alignment with higher timeframe direction.
Works in bull markets via trend following and in bear markets via short signals during downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    # Weekly Supertrend for trend direction
    atr_period = 10
    atr_mult = 3.0
    hl2 = (df_1w['high'] + df_1w['low']) / 2
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift())
    tr3 = abs(df_1w['low'] - df_1w['close'].shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    upper_band = hl2 + (atr_mult * atr)
    lower_band = hl2 - (atr_mult * atr)
    
    supertrend = np.full(len(df_1w), np.nan)
    for i in range(atr_period, len(df_1w)):
        if i == atr_period:
            supertrend[i] = upper_band.iloc[i]
        else:
            if supertrend[i-1] <= upper_band.iloc[i-1]:
                supertrend[i] = min(upper_band.iloc[i], supertrend[i-1])
            else:
                supertrend[i] = max(lower_band.iloc[i], supertrend[i-1])
    
    weekly_trend = np.where(close_1w > supertrend, 1, -1)  # 1: bullish, -1: bearish
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend.astype(float))
    
    # Ichimoku calculations (9, 26, 52)
    conversion_period = 9
    base_period = 26
    lagging_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan_sen = (pd.Series(high).rolling(window=conversion_period, min_periods=conversion_period).max() + 
                  pd.Series(low).rolling(window=conversion_period, min_periods=conversion_period).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun_sen = (pd.Series(high).rolling(window=base_period, min_periods=base_period).max() + 
                 pd.Series(low).rolling(window=base_period, min_periods=base_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward 26 periods
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted forward 26 periods
    senkou_b = ((pd.Series(high).rolling(window=lagging_span_b_period, min_periods=lagging_span_b_period).max() + 
                 pd.Series(low).rolling(window=lagging_span_b_period, min_periods=lagging_span_b_period).min()) / 2)
    
    # Current Kumo (cloud) boundaries - use Senkou Span A and B from 26 periods ago
    senkou_a_shifted = senkou_a.shift(base_period)
    senkou_b_shifted = senkou_b.shift(base_period)
    
    # Upper cloud boundary: max of Senkou A and B
    upper_cloud = np.maximum(senkou_a_shifted, senkou_b_shifted)
    # Lower cloud boundary: min of Senkou A and B
    lower_cloud = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(base_period + lagging_span_b_period, n):  # Start after all indicators are ready
        # Skip if data not ready
        if (np.isnan(tenkan_sen.iloc[i]) or np.isnan(kijun_sen.iloc[i]) or 
            np.isnan(upper_cloud.iloc[i]) or np.isnan(lower_cloud.iloc[i]) or
            np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above cloud, Tenkan > Kijun, weekly trend bullish
            if (close[i] > upper_cloud.iloc[i] and 
                tenkan_sen.iloc[i] > kijun_sen.iloc[i] and 
                weekly_trend_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, Tenkan < Kijun, weekly trend bearish
            elif (close[i] < lower_cloud.iloc[i] and 
                  tenkan_sen.iloc[i] < kijun_sen.iloc[i] and 
                  weekly_trend_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below cloud OR Tenkan < Kijun
                if (close[i] < lower_cloud.iloc[i] or 
                    tenkan_sen.iloc[i] < kijun_sen.iloc[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above cloud OR Tenkan > Kijun
                if (close[i] > upper_cloud.iloc[i] or 
                    tenkan_sen.iloc[i] > kijun_sen.iloc[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0