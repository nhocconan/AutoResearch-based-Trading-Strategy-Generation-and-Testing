#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when Tenkan-sen crosses above Kijun-sen AND price is above the cloud (Senkou Span A/B) 
# AND 1d close > 1d EMA50 AND volume > 2.0 * 20-bar average volume
# Short when Tenkan-sen crosses below Kijun-sen AND price is below the cloud
# AND 1d close < 1d EMA50 AND volume > 2.0 * 20-bar average volume
# Exit when Tenkan-sen crosses back in opposite direction OR price re-enters the cloud
# Ichimoku provides dynamic support/resistance and trend identification
# 1d EMA50 ensures higher timeframe alignment
# Volume confirmation filters low-participation false signals
# Works in bull/bear markets by following the 1d trend while using Ichimoku for precise entries

name = "6h_Ichimoku_TK_Cross_Cloud_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Ichimoku components ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_tenkan = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    low_tenkan = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan_sen = (high_tenkan + low_tenkan) / 2.0
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_kijun = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max()
    low_kijun = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun_sen = (high_kijun + low_kijun) / 2.0
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_senkou_b = pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    low_senkou_b = pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_span_b = ((high_senkou_b + low_senkou_b) / 2.0)
    
    # Align HTF indicators to 6h timeframe (wait for completed 1d bar)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed 1d bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long entry: Tenkan crosses above Kijun AND price above cloud AND uptrend AND volume spike
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and  # Cross just happened
                close[i] > upper_cloud and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Tenkan crosses below Kijun AND price below cloud AND downtrend AND volume spike
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and  # Cross just happened
                  close[i] < lower_cloud and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun OR price re-enters cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]) or \
               (close[i] < upper_cloud and close[i] > lower_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun OR price re-enters cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]) or \
               (close[i] < upper_cloud and close[i] > lower_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals