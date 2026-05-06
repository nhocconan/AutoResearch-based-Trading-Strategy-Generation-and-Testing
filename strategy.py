#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with weekly trend filter
# Long when price breaks above Kumo cloud AND Tenkan-sen > Kijun-sen AND weekly close > weekly EMA50 (bullish regime)
# Short when price breaks below Kumo cloud AND Tenkan-sen < Kijun-sen AND weekly close < weekly EMA50 (bearish regime)
# Exit when price re-enters the Kumo cloud (mean reversion to equilibrium)
# Ichimoku provides dynamic support/resistance; weekly EMA50 ensures higher-timeframe trend alignment
# Designed for 6h timeframe to capture medium-term trends with low trade frequency (target: 12-37 trades/year)
# Works in both bull and bear markets by aligning with weekly regime

name = "6h_Ichimoku_KumoBreakout_WeeklyEMA50_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Kumo cloud boundaries (shifted 26 periods ahead)
    senkou_span_a_shifted = np.roll(senkou_span_a, 26)
    senkou_span_b_shifted = np.roll(senkou_span_b, 26)
    senkou_span_a_shifted[:26] = np.nan
    senkou_span_b_shifted[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_shifted, senkou_span_b_shifted)
    cloud_bottom = np.minimum(senkou_span_a_shifted, senkou_span_b_shifted)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Ichimoku breakout signals with weekly trend filter
            # Long: Price above cloud AND bullish TK cross AND weekly uptrend
            if (close[i] > cloud_top[i] and 
                tenkan_sen[i] > kijun_sen[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud AND bearish TK cross AND weekly downtrend
            elif (close[i] < cloud_bottom[i] and 
                  tenkan_sen[i] < kijun_sen[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price re-enters the cloud (mean reversion)
            if close[i] <= cloud_top[i] and close[i] >= cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price re-enters the cloud (mean reversion)
            if close[i] <= cloud_top[i] and close[i] >= cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals