#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + Kijun/Tenkan Cross with 1d Trend Filter
# Uses Ichimoku on 6h for entry signals (TK cross + price above/below cloud)
# Filters by 1d EMA50 trend (price above/below EMA50) to avoid counter-trend trades
# Works in bull markets (long when price above cloud + uptrend) and bear markets (short when price below cloud + downtrend)
# Target: 50-150 total trades over 4 years = 12-37/year. HARD MAX: 300 total.
# Timeframe: 6h, HTF: 1d

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku parameters (standard)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_high_9 = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    lowest_low_9 = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan_sen = (highest_high_9 + lowest_low_9) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_high_26 = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max()
    lowest_low_26 = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun_sen = (highest_high_26 + lowest_low_26) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted forward 26 periods
    highest_high_52 = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    lowest_low_52 = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_span_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Calculate Chikou Span (Lagging Span): close shifted back 26 periods
    chikou_span = pd.Series(close).shift(-kijun_period)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Ichimoku components and 1d EMA50 to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_span_b.values)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Long entry: TK cross bullish + price above cloud + price above 1d EMA50 (uptrend)
        if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and  # TK cross bullish
            close[i] > cloud_top and                           # Price above cloud
            close[i] > ema_50_1d_aligned[i] and               # Price above 1d EMA50 (uptrend filter)
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: TK cross bearish + price below cloud + price below 1d EMA50 (downtrend)
        elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and  # TK cross bearish
              close[i] < cloud_bottom and                       # Price below cloud
              close[i] < ema_50_1d_aligned[i] and              # Price below 1d EMA50 (downtrend filter)
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: TK cross reverses or price crosses Kijun-sen
        elif position == 1 and (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or close[i] < kijun_sen_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or close[i] > kijun_sen_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dEMA50_Filter"
timeframe = "6h"
leverage = 1.0