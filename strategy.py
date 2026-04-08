#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Weekly Trend Filter + Volume Confirmation
Hypothesis: Ichimoku cloud acts as dynamic support/resistance with forward-looking characteristics. 
Combining with weekly trend filter (price above/below weekly Kumo) and volume confirmation 
creates high-probability trend-following entries that work in both bull and bear markets.
The 6h timeframe targets 15-35 trades/year, avoiding excessive turnover while capturing 
significant moves. Weekly filter prevents counter-trend trades during strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_weekly_trend_volume_v1"
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
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly Ichimoku components for trend filter
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9w = pd.Series(df_weekly['high']).rolling(window=9, min_periods=9).max().values
    low_9w = pd.Series(df_weekly['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9w + low_9w) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26w = pd.Series(df_weekly['high']).rolling(window=26, min_periods=26).max().values
    low_26w = pd.Series(df_weekly['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26w + low_26w) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52w = pd.Series(df_weekly['high']).rolling(window=52, min_periods=52).max().values
    low_52w = pd.Series(df_weekly['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((high_52w + low_52w) / 2)
    
    # Align weekly Ichimoku components to 6h timeframe (with shift for completed bars)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_weekly, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_weekly, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_weekly, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_weekly, senkou_span_b)
    
    # 6h Ichimoku components for entry signals
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_6h = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen_6h = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a_6h = ((tenkan_sen_6h + kijun_sen_6h) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b_6h = ((high_52 + low_52) / 2)
    
    # Volume filter (>1.5x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Senkou Span B calculation
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend: price above/both spans = uptrend, below/both = downtrend
        weekly_uptrend = (close[i] > senkou_span_a_aligned[i] and 
                         close[i] > senkou_span_b_aligned[i])
        weekly_downtrend = (close[i] < senkou_span_a_aligned[i] and 
                           close[i] < senkou_span_b_aligned[i])
        
        # 6h Ichimoku signals
        # TK Cross: Tenkan-sen crossing Kijun-sen
        tk_cross_bull = (tenkan_sen_6h[i] > kijun_sen_6h[i] and 
                        tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1])
        tk_cross_bear = (tenkan_sen_6h[i] < kijun_sen_6h[i] and 
                        tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1])
        
        # Price relative to cloud
        price_above_cloud = (close[i] > senkou_span_a_6h[i] and 
                            close[i] > senkou_span_b_6h[i])
        price_below_cloud = (close[i] < senkou_span_a_6h[i] and 
                            close[i] < senkou_span_b_6h[i])
        
        if position == 1:  # Long position
            # Exit: price closes below cloud or weekly trend turns bearish
            if price_below_cloud or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud or weekly trend turns bullish
            if price_above_cloud or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish TK cross above cloud with weekly uptrend and volume
            if (tk_cross_bull and price_above_cloud and weekly_uptrend and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish TK cross below cloud with weekly downtrend and volume
            elif (tk_cross_bear and price_below_cloud and weekly_downtrend and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals