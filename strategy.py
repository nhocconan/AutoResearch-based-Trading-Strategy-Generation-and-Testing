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
    
    # Load daily data for Ichimoku and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku Cloud components (standard)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    period_kijun = 26
    period_senkou_b = 52
    
    max_high_9 = np.full(len(high_1d), np.nan)
    min_low_9 = np.full(len(low_1d), np.nan)
    max_high_26 = np.full(len(high_1d), np.nan)
    min_low_26 = np.full(len(low_1d), np.nan)
    max_high_52 = np.full(len(high_1d), np.nan)
    min_low_52 = np.full(len(low_1d), np.nan)
    
    for i in range(len(high_1d)):
        if i >= period_tenkan - 1:
            max_high_9[i] = np.max(high_1d[i - period_tenkan + 1:i + 1])
            min_low_9[i] = np.min(low_1d[i - period_tenkan + 1:i + 1])
        if i >= period_kijun - 1:
            max_high_26[i] = np.max(high_1d[i - period_kijun + 1:i + 1])
            min_low_26[i] = np.min(low_1d[i - period_kijun + 1:i + 1])
        if i >= period_senkou_b - 1:
            max_high_52[i] = np.max(high_1d[i - period_senkou_b + 1:i + 1])
            min_low_52[i] = np.min(low_1d[i - period_senkou_b + 1:i + 1])
    
    tenkan_sen = (max_high_9 + min_low_9) / 2
    kijun_sen = (max_high_26 + min_low_26) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate daily ATR for volatility filter
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
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or
            np.isnan(senkou_a_6h[i]) or
            np.isnan(senkou_b_6h[i]) or
            np.isnan(atr_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_6h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        # Determine cloud color (green = bullish, red = bearish)
        cloud_top = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        is_bullish_cloud = senkou_a_6h[i] > senkou_b_6h[i]
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + volume confirmation
            tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
            price_above_cloud = close[i] > cloud_top
            if tk_cross_bullish and price_above_cloud and volume_ratio > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short: TK cross bearish + price below cloud + volume confirmation
            tk_cross_bearish = tenkan_6h[i] < kijun_6h[i]
            price_below_cloud = close[i] < cloud_bottom
            if tk_cross_bearish and price_below_cloud and volume_ratio > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: TK cross bearish OR price falls below cloud
            tk_cross_bearish = tenkan_6h[i] < kijun_6h[i]
            price_below_cloud = close[i] < cloud_bottom
            if tk_cross_bearish or price_below_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: TK cross bullish OR price rises above cloud
            tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
            price_above_cloud = close[i] > cloud_top
            if tk_cross_bullish or price_above_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Ichimoku_TK_Cross_Cloud"
timeframe = "6h"
leverage = 1.0