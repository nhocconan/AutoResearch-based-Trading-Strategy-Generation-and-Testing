#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
Uses 1d EMA50 to determine trend direction (long only when price > EMA50, short only when price < EMA50).
Ichimoku Components (calculated on 6h):
- Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
- Kijun-sen (Base Line): (26-period high + 26-period low)/2
- Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, plotted 26 periods ahead
- Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, plotted 26 periods ahead
- Chikou Span (Lagging Span): Close plotted 26 periods behind
Enter long when: Tenkan > Kijun (bullish TK cross) AND price above cloud (Senkou Span A & B) AND Senkou Span A rising (bullish cloud) AND price > 1d EMA50 (uptrend) AND volume spike.
Enter short when: Tenkan < Kijun (bearish TK cross) AND price below cloud AND Senkou Span A falling (bearish cloud) AND price < 1d EMA50 (downtrend) AND volume spike.
Exit when TK cross reverses or price crosses cloud opposite direction.
Designed for 6h timeframe to maintain 12-35 trades/year. Uses discrete position sizing (0.25) to minimize fee drag.
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
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Ichimoku components on 6h
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (9-period)
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (26-period)
    period_kijun = 26
    max_high_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A: (Tenkan + Kijun)/2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B: (52-period high + low)/2, shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Align all Ichimoku components to 6h timeframe (no additional shift needed as alignment handles completed bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Calculate Senkou Span A slope (1-period change) for cloud direction
    senkou_a_slope = senkou_a_aligned - np.roll(senkou_a_aligned, 1)
    senkou_a_slope[0] = 0
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (need 52 periods for Senkou B)
    start_idx = max(50, 52, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: Bullish TK cross (Tenkan > Kijun) AND price above cloud AND bullish cloud (Senkou A rising) 
            # AND price > 1d EMA50 (uptrend) AND volume spike
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                close[i] > upper_cloud and 
                senkou_a_slope[i] > 0 and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross (Tenkan < Kijun) AND price below cloud AND bearish cloud (Senkou A falling) 
            # AND price < 1d EMA50 (downtrend) AND volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  close[i] < lower_cloud and 
                  senkou_a_slope[i] < 0 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: TK cross reverses OR price crosses cloud in opposite direction
            exit_signal = False
            if position == 1:
                # Exit long when bearish TK cross OR price falls below cloud
                if (tenkan_aligned[i] < kijun_aligned[i] or close[i] < lower_cloud):
                    exit_signal = True
            elif position == -1:
                # Exit short when bullish TK cross OR price rises above cloud
                if (tenkan_aligned[i] > kijun_aligned[i] or close[i] > upper_cloud):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_TK_Cross_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0