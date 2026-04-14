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
    
    # Load 1-day data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = np.full(len(high_1d), np.nan)
    min_low_tenkan = np.full(len(low_1d), np.nan)
    for i in range(period_tenkan - 1, len(high_1d)):
        max_high_tenkan[i] = np.max(high_1d[i - period_tenkan + 1:i + 1])
        min_low_tenkan[i] = np.min(low_1d[i - period_tenkan + 1:i + 1])
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = np.full(len(high_1d), np.nan)
    min_low_kijun = np.full(len(low_1d), np.nan)
    for i in range(period_kijun - 1, len(high_1d)):
        max_high_kijun[i] = np.max(high_1d[i - period_kijun + 1:i + 1])
        min_low_kijun[i] = np.min(low_1d[i - period_kijun + 1:i + 1])
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = np.full(len(high_1d), np.nan)
    min_low_senkou_b = np.full(len(low_1d), np.nan)
    for i in range(period_senkou_b - 1, len(high_1d)):
        max_high_senkou_b[i] = np.max(high_1d[i - period_senkou_b + 1:i + 1])
        min_low_senkou_b[i] = np.min(low_1d[i - period_senkou_b + 1:i + 1])
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe (with proper shift for forward-looking components)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate ATR for volatility filter (14-period)
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(52, n):  # Start after Ichimoku calculation period
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or
            np.isnan(senkou_a_6h[i]) or
            np.isnan(senkou_b_6h[i]) or
            np.isnan(atr_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.8% of price)
        if atr_6h[i] < 0.008 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        # Determine cloud color and position
        # Green cloud: senkou_a > senkou_b (bullish)
        # Red cloud: senkou_a < senkou_b (bearish)
        is_green_cloud = senkou_a_6h[i] > senkou_b_6h[i]
        is_red_cloud = senkou_a_6h[i] < senkou_b_6h[i]
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + volume confirmation
            tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
            price_above_cloud = close[i] > senkou_a_6h[i] and close[i] > senkou_b_6h[i]
            
            if (tk_cross_bullish and 
                price_above_cloud and 
                is_green_cloud and 
                volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: TK cross bearish + price below cloud + volume confirmation
            elif (tenkan_6h[i] < kijun_6h[i] and  # TK cross bearish
                  close[i] < senkou_a_6h[i] and close[i] < senkou_b_6h[i] and  # price below cloud
                  is_red_cloud and  # red cloud
                  volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: TK cross bearish OR price drops below cloud
            tk_cross_bearish = tenkan_6h[i] < kijun_6h[i]
            price_below_cloud = close[i] < senkou_a_6h[i] or close[i] < senkou_b_6h[i]
            
            if tk_cross_bearish or price_below_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: TK cross bullish OR price rises above cloud
            tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
            price_above_cloud = close[i] > senkou_a_6h[i] and close[i] > senkou_b_6h[i]
            
            if tk_cross_bullish or price_above_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Ichimoku_TK_Cross_Cloud_Volume"
timeframe = "6h"
leverage = 1.0