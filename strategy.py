# 6h Ichimoku Cloud + 1d Trend Filter
# Uses 1d Ichimoku for trend direction and 6h for entry timing.
# Long when price above 1d Kumo cloud and 6h TK cross bullish.
# Short when price below 1d Kumo cloud and 6h TK cross bearish.
# Filters out choppy markets and works in both bull/bear regimes.
# Target: 80-120 total trades over 4 years (20-30/year)

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
    
    # === 1d Ichimoku Components ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # === 6h TK Cross (Tenkan/Kijun crossover) ===
    period_tenkan_6h = 9
    period_kijun_6h = 26
    max_high_tenkan_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_tenkan_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_6h = (max_high_tenkan_6h + min_low_tenkan_6h) / 2
    
    max_high_kijun_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_kijun_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_6h = (max_high_kijun_6h + min_low_kijun_6h) / 2
    
    # TK Cross signals
    tk_cross_bullish = tenkan_6h > kijun_6h
    tk_cross_bearish = tenkan_6h < kijun_6h
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for Ichimoku calculations
    warmup = max(52, 26) + 26  # senkou_b period + displacement
    
    for i in range(warmup, n):
        # Skip if any Ichimoku data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Price relative to cloud
        price_above_cloud = price > upper_cloud
        price_below_cloud = price < lower_cloud
        
        # Entry conditions
        if price_above_cloud and tk_cross_bullish[i]:
            # Long: price above cloud and bullish TK cross
            signals[i] = 0.25
        elif price_below_cloud and tk_cross_bearish[i]:
            # Short: price below cloud and bearish TK cross
            signals[i] = -0.25
        else:
            # Hold flat or current signal (no reversal signals to avoid whipsaw)
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_1dCloud_TKCross"
timeframe = "6h"
leverage = 1.0