#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku with 1d cloud filter and volume confirmation
# - Tenkan-sen (9) > Kijun-sen (26) = bullish momentum
# - Price above 1d Kumo (cloud) = bullish trend filter
# - Volume surge confirms breakout strength
# Works in bull/bear via cloud filter (trend direction) + momentum cross
# Target: 50-150 total trades over 4 years (12-37/year)
# Size: 0.25

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 1d Ichimoku components for cloud filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, shifted 26 periods ahead
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2, shifted 26 periods ahead
    senkou_span_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for cloud)
    
    # Align 1d Ichimoku to 6h
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d.values)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)
    
    # 6h Ichimoku for entry signal (Tenkan/Kijun cross)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (9-period)
    tenkan_sen_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (26-period)
    kijun_sen_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    
    # Volume confirmation
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20  # 2x volume surge
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries and color
        span_a = senkou_span_a_1d_aligned[i]
        span_b = senkou_span_b_1d_aligned[i]
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        cloud_bullish = span_a > span_b  # Green cloud when Senkou A > Senkou B
        
        if position == 0:
            # Long: Bullish TK cross + price above cloud + cloud bullish + volume surge
            if (tenkan_sen_6h[i] > kijun_sen_6h[i] and  # Bullish TK cross
                close[i] > cloud_top and                 # Price above cloud
                cloud_bullish and                        # Cloud is bullish
                vol_surge[i]):                           # Volume surge
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross + price below cloud + cloud bearish + volume surge
            elif (tenkan_sen_6h[i] < kijun_sen_6h[i] and  # Bearish TK cross
                  close[i] < cloud_bottom and             # Price below cloud
                  not cloud_bullish and                   # Cloud is bearish
                  vol_surge[i]):                          # Volume surge
                signals[i] = -0.25
                position = -1
        else:
            # Exit: TK cross reverses or price enters cloud
            if position == 1:
                if tenkan_sen_6h[i] <= kijun_sen_6h[i] or close[i] < cloud_top:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if tenkan_sen_6h[i] >= kijun_sen_6h[i] or close[i] > cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0