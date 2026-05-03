#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku TK cross with 1d cloud filter and volume confirmation
# Ichimoku Tenkan-Kijun cross provides momentum signals; trading in direction of 1d cloud (bullish/bearish) 
# filters for higher timeframe trend alignment. Volume confirmation ensures participation.
# Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag.
# Works in bull markets via cloud support/resistance and in bear markets via cloud rejection.

name = "6h_Ichimoku_TK_Cross_1dCloud_Volume"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe (no additional delay needed for cloud)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 6d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after sufficient warmup for Ichimoku
        # Skip if any value is NaN or outside session
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d cloud trend: price above/below cloud
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Long: TK cross bullish (Tenkan > Kijun) with price above cloud and volume spike
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and
                price_above_cloud and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish (Tenkan < Kijun) with price below cloud and volume spike
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and
                  price_below_cloud and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross bearish or price falls below cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]) or \
               close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross bullish or price rises above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]) or \
               close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals