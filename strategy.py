#!/usr/bin/env python3
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
    
    # Load weekly data for Ichimoku components (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly data
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high_weekly).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_weekly).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high_weekly).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_weekly).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2, shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_weekly).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_weekly).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_weekly, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_weekly, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_weekly, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_weekly, senkou_span_b.values)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: TK cross bullish AND price above cloud + volume confirmation
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                close[i] > cloud_top and
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish AND price below cloud + volume confirmation
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  close[i] < cloud_bottom and
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: TK cross reverses OR price returns to cloud
            if position == 1:
                if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                    close[i] < cloud_bottom):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                    close[i] > cloud_top):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Ichimoku_TK_Cross_Cloud_Volume"
timeframe = "6h"
leverage = 1.0