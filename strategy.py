#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm_v2
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter (price > 1d EMA50) and volume confirmation.
Goes long when price breaks above Kumo cloud with bullish TK cross, 1d uptrend, and volume spike.
Short when price breaks below Kumo cloud with bearish TK cross, 1d downtrend, and volume spike.
Exit when price re-enters cloud or TK cross reverses. Uses discrete sizing (0.25) to minimize fees.
Target: 12-30 trades/year. Works in bull via cloud breakouts with trend, in bear via mean reversion at cloud edges.
Added stricter volume confirmation (3x avg) and EMA50 slope filter to reduce false signals.
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
    
    # Get 6h data for Ichimoku calculations
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    kumo_shift = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_6h).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_6h).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_6h).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_6h).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kumo_shift)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = (pd.Series(high_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    senkou_span_b = senkou_span_b.shift(kumo_shift)
    
    # Align Ichimoku components to original timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b.values)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # EMA50 slope filter: only trade when EMA is trending up/down
    ema_50_slope = np.diff(ema_50_1d, prepend=ema_50_1d[0])
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_slope)
    
    # Volume confirmation: volume > 3.0x 20-period average (stricter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (3.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_slope_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Kumo cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        if position == 0:
            # Long: price breaks above cloud, bullish TK cross, 1d uptrend, volume spike, EMA slope up
            long_signal = (close[i] > upper_cloud) and tk_bullish and (close[i] > ema_50_1d_aligned[i]) and vol_spike[i] and (ema_50_slope_aligned[i] > 0)
            # Short: price breaks below cloud, bearish TK cross, 1d downtrend, volume spike, EMA slope down
            short_signal = (close[i] < lower_cloud) and tk_bearish and (close[i] < ema_50_1d_aligned[i]) and vol_spike[i] and (ema_50_slope_aligned[i] < 0)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price re-enters cloud or TK cross turns bearish or EMA slope turns down
            exit_signal = (close[i] <= upper_cloud and close[i] >= lower_cloud) or (not tk_bullish) or (ema_50_slope_aligned[i] <= 0)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price re-enters cloud or TK cross turns bullish or EMA slope turns up
            exit_signal = (close[i] <= upper_cloud and close[i] >= lower_cloud) or (tk_bullish) or (ema_50_slope_aligned[i] >= 0)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm_v2"
timeframe = "6h"
leverage = 1.0