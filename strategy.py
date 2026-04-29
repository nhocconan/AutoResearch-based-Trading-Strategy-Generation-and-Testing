#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Uses Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26, 52) and Chikou Span
# Long when price > cloud AND Tenkan > Kijun AND 1d EMA50 uptrend AND volume spike
# Short when price < cloud AND Tenkan < Kijun AND 1d EMA50 downtrend AND volume spike
# Ichimoku provides dynamic support/resistance and trend direction
# Works in bull markets (trend continuation) and bear markets (counter-trend bounces off cloud)
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_Ichimoku_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components (6h timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Chikou Span (Lagging Shift): close plotted 26 periods behind
    # Not used for entry but can be used for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_senkou_b, period_kijun, period_tenkan, 50)  # 52, 26, 9, 1d EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan_sen[i]
        curr_kijun = kijun_sen[i]
        curr_senkou_a = senkou_span_a[i]
        curr_senkou_b = senkou_span_b[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(curr_senkou_a, curr_senkou_b)
        lower_cloud = min(curr_senkou_a, curr_senkou_b)
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below cloud OR Tenkan < Kijun (momentum loss)
            if curr_close < lower_cloud or curr_tenkan < curr_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above cloud OR Tenkan > Kijun (momentum loss)
            if curr_close > upper_cloud or curr_tenkan > curr_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price > cloud AND Tenkan > Kijun AND 1d EMA50 uptrend AND volume spike
            if (curr_close > upper_cloud and 
                curr_tenkan > curr_kijun and 
                curr_close > curr_ema_1d and  # price above 1d EMA50 for uptrend
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price < cloud AND Tenkan < Kijun AND 1d EMA50 downtrend AND volume spike
            elif (curr_close < lower_cloud and 
                  curr_tenkan < curr_kijun and 
                  curr_close < curr_ema_1d and  # price below 1d EMA50 for downtrend
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals