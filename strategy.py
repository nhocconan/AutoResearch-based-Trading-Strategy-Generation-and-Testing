#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1w trend filter and volume confirmation
# Long when price breaks above weekly Ichimoku cloud AND Tenkan > Kijun AND volume spike
# Short when price breaks below weekly Ichimoku cloud AND Tenkan < Kijun AND volume spike
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Weekly Ichimoku provides strong trend filter that works in both bull (cloud support) and bear (cloud resistance).
# 6h timeframe balances trade frequency and signal quality, avoiding excessive fee drag.

name = "6h_IchimokuCloud_1wTrend_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate weekly Ichimoku components (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need enough data for Ichimoku (26*2)
        return np.zeros(n)
    
    # Ichimoku calculations on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (completed weekly bars only)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 52)  # warmup for volume MA and Ichimoku
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_tenkan = tenkan_sen_aligned[i]
        curr_kijun = kijun_sen_aligned[i]
        curr_span_a = senkou_span_a_aligned[i]
        curr_span_b = senkou_span_b_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        cloud_top = max(curr_span_a, curr_span_b)
        cloud_bottom = min(curr_span_a, curr_span_b)
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above cloud AND Tenkan > Kijun (bullish momentum)
                if (curr_close > cloud_top and 
                    curr_tenkan > curr_kijun):
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below cloud AND Tenkan < Kijun (bearish momentum)
                elif (curr_close < cloud_bottom and 
                      curr_tenkan < curr_kijun):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below cloud bottom (cloud acts as dynamic support)
            if curr_close < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above cloud top (cloud acts as dynamic resistance)
            if curr_close > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals