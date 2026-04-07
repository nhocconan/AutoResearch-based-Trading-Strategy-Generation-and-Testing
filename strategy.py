#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Daily Trend Filter + Volume Confirmation
Hypothesis: Ichimoku system provides multi-component trend confirmation (TK cross, cloud color, price vs cloud).
Using daily timeframe for trend filter and cloud reduces noise. Volume > 1.3x average confirms institutional interest.
Designed for low trade frequency (12-37/year) to avoid fee drag. Works in both bull (cloud as support/resistance) 
and bear (cloud as resistance/support, TK cross signals reversals).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_daily_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                 pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    chikou_span = pd.Series(close).shift(-kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Ichimoku on 6h data
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close, 9, 26, 52)
    
    # Daily trend filter: EMA(21) and price vs cloud
    df_1d = get_htf_data(prices, '1d')
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Daily Ichimoku cloud for trend direction
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, _ = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 9, 26, 52)
    # Cloud top/bottom aligned to 6h
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    
    # Volume filter (>1.3x 50-period average)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Senkou B calculation period
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_21_1d_aligned[i]) or np.isnan(cloud_top_aligned[i]) or 
            np.isnan(cloud_bottom_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud relationship (price above/below cloud)
        price_above_cloud = close[i] > cloud_top_aligned[i]
        price_below_cloud = close[i] < cloud_bottom_aligned[i]
        price_in_cloud = not (price_above_cloud or price_below_cloud)
        
        # TK cross signals
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Daily trend filter: price vs daily cloud and EMA
        daily_uptrend = close[i] > cloud_top_aligned[i] and close[i] > ema_21_1d_aligned[i]
        daily_downtrend = close[i] < cloud_bottom_aligned[i] and close[i] < ema_21_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR TK cross down OR daily trend turns down
            if (price_below_cloud or tk_cross_down or 
                (close[i] < ema_21_1d_aligned[i] and close[i] < cloud_top_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR TK cross up OR daily trend turns up
            if (price_above_cloud or tk_cross_up or 
                (close[i] > ema_21_1d_aligned[i] and close[i] > cloud_bottom_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above cloud + TK cross up + daily uptrend + volume
            if (price_above_cloud and tk_cross_up and daily_uptrend and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price below cloud + TK cross down + daily downtrend + volume
            elif (price_below_cloud and tk_cross_down and daily_downtrend and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals