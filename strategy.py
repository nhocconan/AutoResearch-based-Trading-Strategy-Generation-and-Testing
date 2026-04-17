#!/usr/bin/env python3
"""
Hypothesis: On 12h timeframe, price tends to respect weekly Ichimoku Cloud support/resistance.
Combining weekly Ichimoku Cloud with 12h volume spikes and price above/below Kumo creates
high-probability trend-following trades. The weekly Cloud acts as dynamic support/resistance
while volume confirms breakout strength. Strategy targets 15-25 trades per year by requiring
price to break above/below Cloud with volume confirmation. Works in bull markets (trend
continuation above Cloud) and bear markets (trend continuation below Cloud).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Ichimoku Cloud
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku Cloud components (weekly)
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
    period52_high = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 12h timeframe
    tenkan_12h = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_12h = align_htf_to_ltf(prices, df_1w, kijun_sen)
    span_a_12h = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    span_b_12h = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Volume confirmation: 20-period volume MA on 12h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for Ichimoku calculations
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_12h[i]) or np.isnan(kijun_12h[i]) or 
            np.isnan(span_a_12h[i]) or np.isnan(span_b_12h[i]) or
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        # Determine Cloud boundaries and trend
        upper_cloud = max(span_a_12h[i], span_b_12h[i])
        lower_cloud = min(span_a_12h[i], span_b_12h[i])
        in_cloud = lower_cloud <= price <= upper_cloud
        above_cloud = price > upper_cloud
        below_cloud = price < lower_cloud
        
        if position == 0:
            # Long: price breaks above Cloud with volume spike
            if above_cloud and vol > 1.8 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Cloud with volume spike
            elif below_cloud and vol > 1.8 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price re-enters Cloud or volume drops
            if in_cloud or vol < 0.7 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price re-enters Cloud or volume drops
            if in_cloud or vol < 0.7 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_IchimokuCloud_Volume_Breakout"
timeframe = "12h"
leverage = 1.0