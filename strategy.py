#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud system with 12-hour trend filter
# Uses Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h data
# Trend filter from 12h EMA cross to avoid counter-trend trades
# Only trades when price is above/both Kumo (cloud) with TK cross in same direction
# Designed to work in both bull and bear markets via trend filter and cloud support/resistance
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_ichimoku_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(25) and EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Ichimoku calculations (9, 26, 52 periods)
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
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals as it requires future data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema_25_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine if price is above or below cloud
        # Cloud top is the higher of Senkou Span A and B
        # Cloud bottom is the lower of Senkou Span A and B
        cloud_top = np.maximum(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = np.minimum(senkou_span_a[i], senkou_span_b[i])
        
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK Cross: Tenkan-sen crossing Kijun-sen
        tk_cross_up = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_down = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        # Trend filter from 12h EMA
        uptrend_12h = ema_25_12h_aligned[i] > ema_50_12h_aligned[i]
        downtrend_12h = ema_25_12h_aligned[i] < ema_50_12h_aligned[i]
        
        if position == 1:  # long position
            # Exit: price closes below cloud or TK cross down
            if close[i] < cloud_bottom or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above cloud or TK cross up
            if close[i] > cloud_top or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Long: price above cloud + TK cross up + 12h uptrend
            if price_above_cloud and tk_cross_up and uptrend_12h:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross down + 12h downtrend
            elif price_below_cloud and tk_cross_down and downtrend_12h:
                signals[i] = -0.25
                position = -1
    
    return signals