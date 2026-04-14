#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-day Tenkan/Kijun cross filter and volume confirmation.
# The Ichimoku Cloud provides dynamic support/resistance and trend direction.
# Price above/below cloud indicates bullish/bearish bias.
# Tenkan/Kijun cross signals momentum shifts in the direction of the cloud.
# Volume > 1.3x average confirms institutional participation.
# This strategy aims for 15-30 trades per year per symbol (60-120 total over 4 years),
# staying within optimal range to minimize fee drag while capturing major trends.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    if len(df_1d) < senkou_span_b_period:
        return np.zeros(n)
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(df_1d['high']).rolling(window=tenkan_period, min_periods=tenkan_period).max() +
                  pd.Series(df_1d['low']).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(df_1d['high']).rolling(window=kijun_period, min_periods=kijun_period).max() +
                 pd.Series(df_1d['low']).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(df_1d['high']).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() +
                     pd.Series(df_1d['low']).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, senkou_span_b_period)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries and trend
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Price above/below cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # Tenkan/Kijun cross
        tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price above cloud + TK cross up + volume
            if (price_above_cloud and 
                tk_cross_up and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price below cloud + TK cross down + volume
            elif (price_below_cloud and 
                  tk_cross_down and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price drops below cloud or TK cross down
            if (close[i] < lower_cloud or 
                tk_cross_down):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises above cloud or TK cross up
            if (close[i] > upper_cloud or 
                tk_cross_up):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Ichimoku_TK_Cross_Volume_v1"
timeframe = "6h"
leverage = 1.0