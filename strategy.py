#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_v1
Hypothesis: Trade Ichimoku cloud breaks in direction of 12h Kumo twist (Senkou Span A/B cross) with volume confirmation. Works in bull markets (long when price above cloud + bullish Kumo twist) and bear markets (short when price below cloud + bearish Kumo twist). Targets 15-30 trades/year by requiring confluence: price break of cloud + Kumo twist alignment + volume spike (>1.8x average). Uses ATR trailing stop (2.0). Designed for 6h timeframe to avoid overtrading while capturing medium-term trends.
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
    
    # Get 6h data for Ichimoku calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need at least 52 periods for Ichimoku
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
    
    # Get 12h data for HTF Kumo twist filter (Senkou Span A/B cross)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 12h data for Kumo twist
    period9_high_12h = pd.Series(df_12h['high']).rolling(window=9, min_periods=9).max().values
    period9_low_12h = pd.Series(df_12h['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen_12h = (period9_high_12h + period9_low_12h) / 2
    
    period26_high_12h = pd.Series(df_12h['high']).rolling(window=26, min_periods=26).max().values
    period26_low_12h = pd.Series(df_12h['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen_12h = (period26_high_12h + period26_low_12h) / 2
    
    senkou_span_a_12h = ((tenkan_sen_12h + kijun_sen_12h) / 2)
    
    period52_high_12h = pd.Series(df_12h['high']).rolling(window=52, min_periods=52).max().values
    period52_low_12h = pd.Series(df_12h['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b_12h = ((period52_high_12h + period52_low_12h) / 2)
    
    # Kumo twist: Senkou Span A crossing above/below Senkou Span B
    # Bullish twist: Senkou Span A > Senkou Span B
    # Bearish twist: Senkou Span A < Senkou Span B
    kumo_twist_bullish = senkou_span_a_12h > senkou_span_b_12h
    kumo_twist_bearish = senkou_span_a_12h < senkou_span_b_12h
    
    # Align Ichimoku components and Kumo twist to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_12h, kumo_twist_bullish.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_12h, kumo_twist_bearish.astype(float))
    
    # Volume confirmation: 1.8x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop (14-period on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of Ichimoku (52), volume MA (20), ATR (14)
    start_idx = max(52, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(kumo_twist_bullish_aligned[i]) or 
            np.isnan(kumo_twist_bearish_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        span_a_val = senkou_span_a_aligned[i]
        span_b_val = senkou_span_b_aligned[i]
        kumo_twist_bull = kumo_twist_bullish_aligned[i] > 0.5
        kumo_twist_bear = kumo_twist_bearish_aligned[i] > 0.5
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_14_val = atr_14[i]
        
        # Determine cloud boundaries (upper and lower band of Kumo)
        cloud_top = max(span_a_val, span_b_val)
        cloud_bottom = min(span_a_val, span_b_val)
        
        if position == 0:
            # Long: price breaks above cloud, bullish Kumo twist, volume spike
            long_signal = (close_val > cloud_top) and kumo_twist_bull and (volume_val > 1.8 * vol_ma_val)
            # Short: price breaks below cloud, bearish Kumo twist, volume spike
            short_signal = (close_val < cloud_bottom) and kumo_twist_bear and (volume_val > 1.8 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or price breaks below cloud
            if (low_val < long_stop) or (close_val < cloud_bottom):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or price breaks above cloud
            if (high_val > short_stop) or (close_val > cloud_top):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_v1"
timeframe = "6h"
leverage = 1.0