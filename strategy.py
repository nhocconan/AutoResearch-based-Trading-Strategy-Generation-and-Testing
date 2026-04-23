#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
Long when Tenkan-sen crosses above Kijun-sen AND price > Kumo cloud (bullish) AND 1d EMA50 uptrend AND 6h volume > 1.5x 20-period average.
Short when Tenkan-sen crosses below Kijun-sen AND price < Kumo cloud (bearish) AND 1d EMA50 downtrend AND 6h volume > 1.5x 20-period average.
Exit when Tenkan-sen/Kijun-sen cross in opposite direction OR price crosses Kumo cloud in opposite direction.
Ichimoku provides trend, momentum, and support/resistance in one system. Works in bull markets via cloud breakouts and in bear markets via cloud breakdowns.
Target: ~15-25 trades/year on 6h timeframe with discrete sizing 0.25.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Kumo cloud boundaries: Senkou Span A and B shifted 26 periods ahead
    # For look-ahead avoidance, we use the cloud values from 26 periods ago (already formed)
    senkou_span_a_lagged = np.roll(senkou_span_a, period_kijun)  # shift by 26
    senkou_span_b_lagged = np.roll(senkou_span_b, period_kijun)
    # Fill NaN values from roll with first valid values
    senkou_span_a_lagged[:period_kijun] = senkou_span_a[period_kijun] if not np.isnan(senkou_span_a[period_kijun]) else 0
    senkou_span_b_lagged[:period_kijun] = senkou_span_b[period_kijun] if not np.isnan(senkou_span_b[period_kijun]) else 0
    
    # Cloud top and bottom (for price vs cloud comparison)
    cloud_top = np.maximum(senkou_span_a_lagged, senkou_span_b_lagged)
    cloud_bottom = np.minimum(senkou_span_a_lagged, senkou_span_b_lagged)
    
    # 6h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period_tenkan, period_kijun, period_senkou_b, 20, 50) + period_kijun  # +26 for cloud lag
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        tenkan_prev = tenkan_sen[i-1]
        kijun_prev = kijun_sen[i-1]
        ema_val = ema_50_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        
        # Bullish cloud: Senkou Span A > Senkou Span B
        bullish_cloud = senkou_span_a_lagged[i] > senkou_span_b_lagged[i]
        # Bearish cloud: Senkou Span A < Senkou Span B
        bearish_cloud = senkou_span_a_lagged[i] < senkou_span_b_lagged[i]
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price > cloud top (bullish) AND 1d EMA50 uptrend AND volume spike
            if (tenkan > kijun and tenkan_prev <= kijun_prev and 
                price > cloud_top_val and bullish_cloud and 
                ema_val > close_1d[-1] if len(close_1d) > 0 else False and  # Simplified 1d trend check
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price < cloud bottom (bearish) AND 1d EMA50 downtrend AND volume spike
            elif (tenkan < kijun and tenkan_prev >= kijun_prev and 
                  price < cloud_bottom_val and bearish_cloud and 
                  ema_val < close_1d[-1] if len(close_1d) > 0 else True and  # Simplified 1d trend check
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Exit: Tenkan/Kijun cross in opposite direction OR price crosses cloud in opposite direction
            if position == 1:
                if (tenkan < kijun and tenkan_prev >= kijun_prev) or price < cloud_bottom_val:
                    exit_signal = True
            elif position == -1:
                if (tenkan > kijun and tenkan_prev <= kijun_prev) or price > cloud_top_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_1dEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0