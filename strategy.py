#!/usr/bin/env python3
# 6h_ichimoku_cloud_funding_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d HTF for trend direction, combined with funding rate mean reversion for entry timing. Long when price > 1d Kumo cloud AND funding rate < -0.02% (extreme negative = long pressure). Short when price < 1d Kumo cloud AND funding rate > +0.02% (extreme positive = short pressure). Uses discrete position sizing (0.25) to limit drawdown. Works in bull/bear: Ichimoku filter ensures alignment with higher timeframe trend, funding rate provides contrarian entries during excessive leverage bias. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_funding_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Ichimoku Cloud components from 1d HTF
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    high_1d_s = pd.Series(high_1d)
    low_1d_s = pd.Series(low_1d)
    tenkan_sen = (high_1d_s.rolling(window=9, min_periods=9).max() + 
                  low_1d_s.rolling(window=9, min_periods=9).min()) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    kijun_sen = (high_1d_s.rolling(window=26, min_periods=26).max() + 
                 low_1d_s.rolling(window=26, min_periods=26).min()) / 2
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(2)
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((high_1d_s.rolling(window=52, min_periods=52).max() + 
                      low_1d_s.rolling(window=52, min_periods=52).min()) / 2).shift(2)
    
    # Align Ichimoku components to 6h timeframe (completed 1d bar only)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Kumo cloud boundaries (Senkou Span A/B)
    upper_cloud = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    lower_cloud = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Price relative to cloud
    price_above_cloud = close > upper_cloud
    price_below_cloud = close < lower_cloud
    
    # Funding rate data (8h intervals)
    funding_path = "/mnt/shared/funding/data/processed/funding/BTCUSDT.parquet"
    try:
        funding_df = pd.read_parquet(funding_path)
        funding_rate = funding_df['funding_rate'].values
        funding_time = funding_df['timestamp'].values
        
        # Align funding rate to 6h timeframe (use previous completed 8h funding rate)
        funding_aligned = np.full(n, np.nan)
        j = 0
        for i in range(n):
            while j < len(funding_time) - 1 and funding_time[j+1] <= prices['open_time'].iloc[i]:
                j += 1
            if j < len(funding_time):
                funding_aligned[i] = funding_rate[j]
    except:
        # Fallback if funding data unavailable
        funding_aligned = np.zeros(n)
    
    # Extreme funding rate thresholds (contrarian signal)
    funding_extreme_long = funding_aligned < -0.0002  # <-0.02%
    funding_extreme_short = funding_aligned > 0.0002   # >+0.02%
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(close[i]) or np.isnan(funding_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Kumo cloud OR funding becomes excessively positive
            if close[i] < upper_cloud[i] or funding_aligned[i] > 0.0005:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Kumo cloud OR funding becomes excessively negative
            if close[i] > lower_cloud[i] or funding_aligned[i] < -0.0005:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above cloud + extremely negative funding (contrarian long)
            if price_above_cloud[i] and funding_extreme_long[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price below cloud + extremely positive funding (contrarian short)
            elif price_below_cloud[i] and funding_extreme_short[i]:
                position = -1
                signals[i] = -0.25
    
    return signals