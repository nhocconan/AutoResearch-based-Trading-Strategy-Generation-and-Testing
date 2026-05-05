#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud Breakout with Weekly Trend Filter
# Long when: Tenkan-sen crosses above Kijun-sen AND price is above Kumo (cloud) AND weekly close > weekly open (bullish weekly candle)
# Short when: Tenkan-sen crosses below Kijun-sen AND price is below Kumo (cloud) AND weekly close < weekly open (bearish weekly candle)
# Exit when: Tenkan-sen crosses back in opposite direction OR price re-enters the cloud
# Ichimoku provides multiple confirmation lines (Tenkan, Kijun, Senkou Span A/B) reducing false breakouts
# Weekly trend filter ensures we only trade in the direction of the higher timeframe momentum
# Works in both bull and bear markets by aligning with weekly momentum while using 6h for precise entry/exit
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_IchimokuCloud_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish weekly candle
    weekly_bearish = weekly_close < weekly_open  # True for bearish weekly candle
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted 26 periods ahead
    # For lookahead safety, we use the cloud values from 26 periods ago to represent current cloud
    senkou_span_a_lag = np.roll(senkou_span_a, 26)
    senkou_span_b_lag = np.roll(senkou_span_b, 26)
    senkou_span_a_lag[:26] = np.nan
    senkou_span_b_lag[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_lag, senkou_span_b_lag)
    cloud_bottom = np.minimum(senkou_span_a_lag, senkou_span_b_lag)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish weekly trend filter
        weekly_trend_up = weekly_bullish_aligned[i] > 0.5
        weekly_trend_down = weekly_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud AND weekly bullish
            if (tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1] and
                close[i] > cloud_top[i] and weekly_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below cloud AND weekly bearish
            elif (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1] and
                  close[i] < cloud_bottom[i] and weekly_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun OR price re-enters cloud
            if (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]) or \
               (close[i] < cloud_top[i] and close[i] > cloud_bottom[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun OR price re-enters cloud
            if (tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]) or \
               (close[i] < cloud_top[i] and close[i] > cloud_bottom[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals