#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Uses Ichimoku (Tenkan, Kijun, Senkou Span A/B, Chikou) for trend identification and momentum
# Long when price > cloud AND Tenkan > Kijun AND Chikou > price 26 periods ago AND 1d ADX > 25 AND volume > 1.5x 20 EMA
# Short when price < cloud AND Tenkan < Kijun AND Chikou < price 26 periods ago AND 1d ADX > 25 AND volume > 1.5x 20 EMA
# Ichimoku provides multiple confirmation layers reducing false signals. 6h timeframe avoids overtrading.
# Designed for 12-37 trades/year with discrete sizing (0.25). Works in bull/bear via trend following.

name = "6h_Ichimoku_1dADX_VolumeConfirm"
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
    
    # Get 1d data for HTF ADX filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_trending = adx > 25  # Strong trend filter
    
    # Align 1d ADX trend to 6h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    chikou_shift = 26
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_high_9 = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_low_9 = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_high_26 = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_low_26 = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted 26 periods ahead
    highest_high_52 = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_low_52 = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    chikou = np.roll(close, chikou_shift)
    chikou[:chikou_shift] = np.nan  # First 26 values are invalid
    
    # Current cloud boundaries (Senkou Span A/B from 26 periods ago)
    senkou_span_a_lag = np.roll(senkou_span_a, chikou_shift)
    senkou_span_b_lag = np.roll(senkou_span_b, chikou_shift)
    senkou_span_a_lag[:chikou_shift] = np.nan
    senkou_span_b_lag[:chikou_shift] = np.nan
    
    # Identify cloud (top and bottom of cloud)
    cloud_top = np.maximum(senkou_span_a_lag, senkou_span_b_lag)
    cloud_bottom = np.minimum(senkou_span_a_lag, senkou_span_b_lag)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_trending_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(chikou[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Price > cloud AND Tenkan > Kijun AND Chikou > price 26 periods ago AND 1d trending AND volume spike
            if (close[i] > cloud_top[i] and 
                tenkan[i] > kijun[i] and 
                chikou[i] > close[i - chikou_shift] if i >= chikou_shift else False and 
                adx_trending_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Price < cloud AND Tenkan < Kijun AND Chikou < price 26 periods ago AND 1d trending AND volume spike
            elif (close[i] < cloud_bottom[i] and 
                  tenkan[i] < kijun[i] and 
                  chikou[i] < close[i - chikou_shift] if i >= chikou_shift else False and 
                  adx_trending_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price < cloud OR Tenkan < Kijun OR 1d trend weakens
            if (close[i] < cloud_top[i] or 
                tenkan[i] < kijun[i] or 
                adx_trending_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price > cloud OR Tenkan > Kijun OR 1d trend weakens
            if (close[i] > cloud_bottom[i] or 
                tenkan[i] > kijun[i] or 
                adx_trending_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals