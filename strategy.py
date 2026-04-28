#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku cloud breakout with TK cross confirmation and volume filter.
# Enter long when price breaks above 1d Ichimoku cloud (Senkou Span A/B) with TK cross bullish and volume > 1.5x average.
# Enter short when price breaks below 1d Ichimoku cloud with TK cross bearish and volume > 1.5x average.
# Uses discrete position sizing (0.25) to manage drawdown. Target: 12-30 trades/year.
# Ichimoku provides dynamic support/resistance from higher timeframe, TK cross confirms momentum, volume validates breakout.
# Works in bull (breakouts with trend) and bear (failed breaks reverse via exits) markets.

name = "6h_Ichimoku_Cloud_TK_Cross_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = np.full(n_1d, np.nan)
    for i in range(period_tenkan - 1, n_1d):
        window_high = np.max(high_1d[i - period_tenkan + 1:i + 1])
        window_low = np.min(low_1d[i - period_tenkan + 1:i + 1])
        tenkan_sen[i] = (window_high + window_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = np.full(n_1d, np.nan)
    for i in range(period_kijun - 1, n_1d):
        window_high = np.max(high_1d[i - period_kijun + 1:i + 1])
        window_low = np.min(low_1d[i - period_kijun + 1:i + 1])
        kijun_sen[i] = (window_high + window_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = np.full(n_1d, np.nan)
    for i in range(n_1d):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = np.full(n_1d, np.nan)
    for i in range(period_senkou_b - 1, n_1d):
        window_high = np.max(high_1d[i - period_senkou_b + 1:i + 1])
        window_low = np.min(low_1d[i - period_senkou_b + 1:i + 1])
        senkou_span_b[i] = (window_high + window_low) / 2.0
    
    # Shift Senkou Spans forward by 26 periods (for cloud plotting)
    senkou_span_a_shifted = np.full(n_1d, np.nan)
    senkou_span_b_shifted = np.full(n_1d, np.nan)
    for i in range(n_1d - 26):
        senkou_span_a_shifted[i + 26] = senkou_span_a[i]
        senkou_span_b_shifted[i + 26] = senkou_span_b[i]
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_shifted)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_shifted)
    
    # Calculate 6h volume filter: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross conditions
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Price breakout conditions
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Entry conditions: price breaks cloud with TK cross alignment and volume filter
        long_entry = price_above_cloud and tk_bullish and volume_filter[i]
        short_entry = price_below_cloud and tk_bearish and volume_filter[i]
        
        # Exit conditions: price returns to cloud or TK cross reverses
        long_exit = close[i] < cloud_bottom or (position == 1 and not tk_bullish)
        short_exit = close[i] > cloud_top or (position == -1 and not tk_bearish)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals