#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Ichimoku provides multi-component trend system: Tenkan/Kijun cross for momentum,
# Senkou Span A/B for dynamic support/resistance (cloud), Chikou for confirmation
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# Volume spike (2.0x 20-period average) confirms institutional participation
# Works in both bull and bear: trend filter prevents counter-trend trades, cloud acts as dynamic S/R
# Targets 12-37 trades per year (50-150 total over 4 years) to minimize fee drag

name = "6h_Ichimoku_Cloud_1dEMA50_VolumeSpike"
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
    
    # Load 6h data ONCE before loop for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need enough for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_6h['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_6h['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_6h['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_6h['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_6h['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_6h['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = pd.Series(df_6h['close']).shift(26).values
    
    # Align Ichimoku components to 6h timeframe (already on 6h, so direct use)
    tenkan_sen_aligned = tenkan_sen
    kijun_sen_aligned = kijun_sen
    senkou_span_a_aligned = senkou_span_a
    senkou_span_b_aligned = senkou_span_b
    chikou_span_aligned = chikou_span
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Ichimoku: 52 + 26 shift)
    start_idx = 78
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(chikou_span_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            # Long: Price above cloud AND Tenkan > Kijun AND price > 1d EMA50 AND volume spike
            if (close[i] > cloud_top and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud AND Tenkan < Kijun AND price < 1d EMA50 AND volume spike
            elif (close[i] < cloud_bottom and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below cloud OR Tenkan < Kijun OR price < 1d EMA50
            if (close[i] < cloud_bottom or 
                tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above cloud OR Tenkan > Kijun OR price > 1d EMA50
            if (close[i] > cloud_top or 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals