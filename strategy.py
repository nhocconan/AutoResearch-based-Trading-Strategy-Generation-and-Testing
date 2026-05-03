#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud Breakout with 1d volume spike and 12h ADX trend filter
# Ichimoku provides dynamic support/resistance via cloud (Senkou Span A/B) and momentum via TK cross.
# Breakouts above/below cloud with volume confirmation and 12h ADX>25 trend filter capture
# strong momentum moves while avoiding whipsaws in ranging markets. Designed for low trade
# frequency (12-37/year) on 6h timeframe to minimize fee drag. Works in bull/bear markets by
# trading breakouts in direction of higher timeframe trend.

name = "6h_Ichimoku_CloudBreakout_12hADX25_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = df_1d['close'].values.copy()
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = pd.Series(df_12h['high'].values).diff().abs()
    tr2 = (pd.Series(df_12h['high'].values) - pd.Series(df_12h['close'].values.shift(1))).abs()
    tr3 = (pd.Series(df_12h['low'].values) - pd.Series(df_12h['close'].values.shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(df_12h['high'].values).diff()
    dm_minus = -pd.Series(df_12h['low'].values).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Get 1d data for volume spike
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Determine trend direction from 12h ADX
        is_strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above cloud with TK bullish cross, volume spike, and strong trend
            tk_bullish = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            if close[i] > upper_cloud and tk_bullish and volume_spike_aligned[i] and is_strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud with TK bearish cross, volume spike, and strong trend
            elif close[i] < lower_cloud and not tk_bullish and tenkan_aligned[i] < kijun_aligned[i] and volume_spike_aligned[i] and is_strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters cloud or TK bearish cross
            if close[i] < lower_cloud or (tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters above cloud or TK bullish cross
            if close[i] > upper_cloud or (tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals