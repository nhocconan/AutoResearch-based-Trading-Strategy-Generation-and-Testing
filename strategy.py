#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Ichimoku components (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:  # Need enough for Ichimoku (26*2)
        return np.zeros(n)
    
    # Ichimoku components on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): close shifted back 26 periods (not used for signals)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: 24-period average on 6h (equivalent to ~6d on 1d)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: TK cross above AND price above cloud WITH volume spike
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                close[i] > cloud_top and 
                volume[i] > 2.0 * vol_avg_24[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross below AND price below cloud WITH volume spike
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  close[i] < cloud_bottom and 
                  volume[i] > 2.0 * vol_avg_24[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: TK cross in opposite direction OR price re-enters cloud
            if position == 1:
                # Exit long: TK cross below OR price below cloud top
                if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                    close[i] < cloud_top):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: TK cross above OR price above cloud bottom
                if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                    close[i] > cloud_bottom):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Ichimoku_TK_Cross_Cloud_Filter_Volume"
timeframe = "6h"
leverage = 1.0