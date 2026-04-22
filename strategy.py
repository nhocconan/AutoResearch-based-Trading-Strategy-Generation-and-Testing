#!/usr/bin/env python3

"""
Hypothesis: 6-hour Ichimoku Cloud breakout with 1-day trend filter and volume confirmation.
The Ichimoku Cloud provides dynamic support/resistance and trend direction.
The 1-day trend filter ensures trades align with the daily trend to avoid counter-trend trades.
Volume spikes confirm institutional participation at breakout points.
This strategy aims to capture strong momentum moves in both bull and bear markets by
trading breakouts of the Kumo (cloud) with trend and volume confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                 pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h Ichimoku data - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku on 6h data
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(
        df_6h['high'].values, df_6h['low'].values, df_6h['close'].values
    )
    
    # Align Ichimoku components to 6h timeframe (already aligned, but for safety)
    tenkan_6h_aligned = align_htf_to_ltf(prices, df_6h, tenkan_6h)
    kijun_6h_aligned = align_htf_to_ltf(prices, df_6h, kijun_6h)
    senkou_a_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_a_6h)
    senkou_b_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_b_6h)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume average (24-period)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(tenkan_6h_aligned[i]) or np.isnan(kijun_6h_aligned[i]) or 
            np.isnan(senkou_a_6h_aligned[i]) or np.isnan(senkou_b_6h_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud color and position
        # Green cloud (bullish): Senkou A > Senkou B
        # Red cloud (bearish): Senkou A < Senkou B
        cloud_top = np.maximum(senkou_a_6h_aligned[i], senkou_b_6h_aligned[i])
        cloud_bottom = np.minimum(senkou_a_6h_aligned[i], senkou_b_6h_aligned[i])
        is_bullish_cloud = senkou_a_6h_aligned[i] > senkou_b_6h_aligned[i]
        
        if position == 0:
            # Long: price breaks above cloud, bullish cloud, above 1d EMA, volume spike
            if (close[i] > cloud_top and                    # Price above cloud
                is_bullish_cloud and                        # Bullish cloud
                close[i] > ema_34_1d_aligned[i] and         # Above 1d EMA (bullish trend)
                volume[i] > 2.0 * vol_avg_24[i]):           # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud, bearish cloud, below 1d EMA, volume spike
            elif (close[i] < cloud_bottom and               # Price below cloud
                  not is_bullish_cloud and                  # Bearish cloud
                  close[i] < ema_34_1d_aligned[i] and       # Below 1d EMA (bearish trend)
                  volume[i] > 2.0 * vol_avg_24[i]):         # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite side of cloud or crosses 1d EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below cloud or below 1d EMA
                if close[i] < cloud_bottom or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above cloud or above 1d EMA
                if close[i] > cloud_top or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0