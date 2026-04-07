#!/usr/bin/env python3
"""
6h Ichimoku Cloud with Daily Trend Filter
Long when Tenkan > Kijun and price above cloud with bullish daily bias
Short when Tenkan < Kijun and price below cloud with bearish daily bias
Exit when price crosses Tenkan-Kijun line or daily trend flips
Ichimoku provides dynamic support/resistance and trend direction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Ichimoku Components ===
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max()
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    low_senkou = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b = ((high_senkou + low_senkou) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    # Not used in signals to avoid look-ahead
    
    # === Daily Trend Filter (1D) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA 50 for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily trend to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily trend: 1 = bullish (price above EMA50), -1 = bearish (price below EMA50)
    daily_trend = np.where(close_1d > ema_50_1d, 1, -1)
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after Ichimoku warmup
        # Skip if any values are NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 1:  # Long position
            # Exit: price crosses below Tenkan-Kijun midpoint OR daily trend turns bearish
            tk_mid = (tenkan[i] + kijun[i]) / 2
            if close[i] < tk_mid or daily_trend_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Tenkan-Kijun midpoint OR daily trend turns bullish
            tk_mid = (tenkan[i] + kijun[i]) / 2
            if close[i] > tk_mid or daily_trend_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need price clearly above/below cloud
            tk_mid = (tenkan[i] + kijun[i]) / 2
            
            # Long: price above cloud, bullish TK cross, bullish daily trend
            if (close[i] > cloud_top and 
                tenkan[i] > kijun[i] and 
                daily_trend_aligned[i] == 1):
                position = 1
                signals[i] = 0.25
            
            # Short: price below cloud, bearish TK cross, bearish daily trend
            elif (close[i] < cloud_bottom and 
                  tenkan[i] < kijun[i] and 
                  daily_trend_aligned[i] == -1):
                position = -1
                signals[i] = -0.25
    
    return signals