#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_Volume
Hypothesis: Uses 6h timeframe with Ichimoku cloud breakout filtered by daily trend and volume spikes.
Ichimoku provides multiple confirmation lines (Tenkan, Kijun, Senkou Span A/B, Chikou).
Only take trades when price breaks above/below cloud in direction of daily trend with volume confirmation.
Works in both bull and bear markets by only taking trades in direction of daily trend.
"""

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = np.concatenate([np.full(26, np.nan), close[:-26]]) if len(close) > 26 else np.full_like(close, np.nan)
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate Ichimoku on 6h data using current bar's data (no look-ahead as we use current values)
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # For cloud top/bottom, we need to shift Senkou spans forward by 26 periods
    # But since we're using current values, we'll use the current Senkou spans as the cloud
    # The cloud is actually formed by Senkou A and B plotted 26 periods ahead
    # So for current price, we compare against Senkou A/B from 26 periods ago
    senkou_a_lagged = np.concatenate([np.full(26, np.nan), senkou_a[:-26]]) if len(senkou_a) > 26 else np.full_like(senkou_a, np.nan)
    senkou_b_lagged = np.concatenate([np.full(26, np.nan), senkou_b[:-26]]) if len(senkou_b) > 26 else np.full_like(senkou_b, np.nan)
    
    # Cloud top is max of Senkou A/B, cloud bottom is min
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) for spike detection
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start from 52 to have enough data for Ichimoku
        # Get aligned values for current 6h bar
        ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)[i]
        vol_avg_val = vol_avg_20[i]
        
        # Skip if any required data is NaN
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(ema50_aligned) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above cloud + daily uptrend + volume spike
            if (close[i] > cloud_top[i] and 
                close[i] > ema50_aligned and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below cloud + daily downtrend + volume spike
            elif (close[i] < cloud_bottom[i] and 
                  close[i] < ema50_aligned and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below cloud or trend turns down
            if (close[i] < cloud_top[i] or close[i] < ema50_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above cloud or trend turns up
            if (close[i] > cloud_bottom[i] or close[i] > ema50_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals