#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_AvoidChop
Hypothesis: Trade 4h Camarilla R3/S3 breakouts aligned with 12h EMA50 trend, with volume spike confirmation and choppiness filter to avoid ranging markets.
Designed to work in both bull and bear markets via 12h trend filter + volume conviction + regime avoidance.
Target: 20-30 trades/year per symbol (<100 total 4h trades over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for price action and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from prior 12h OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: R3/S3 (deeper retracement = stronger breakout signal)
    range_12h = high_12h - low_12h
    camarilla_pivot = (high_12h + low_12h + close_12h) / 3
    camarilla_r3 = camarilla_pivot + (range_12h * 3.0 / 12)
    camarilla_s3 = camarilla_pivot - (range_12h * 3.0 / 12)
    
    # Align Camarilla levels to 4h timeframe (prior 12h's levels available at 12h close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume confirmation: volume > 1.8x 20-period average on 4h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    # Choppiness Index regime filter (using 4h data) - avoid choppy markets
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index
    chop = np.where(
        (sum_atr_14 > 0) & (range_14 > 0),
        100 * np.log10(sum_atr_14 / range_14) / np.log10(14),
        50
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), volume MA (20), ATR (14+14), chop (14+14)
    start_idx = max(50, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_ma[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Regime filter: avoid strong ranging markets (chop > 55)
        not_ranging = chop[i] < 55
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + price above 12h EMA50 + volume spike + not ranging
            long_breakout = close[i] > camarilla_r3_aligned[i]
            long_signal = long_breakout and price_above_ema and volume_spike[i] and not_ranging
            
            # Short: price breaks below Camarilla S3 + price below 12h EMA50 + volume spike + not ranging
            short_breakout = close[i] < camarilla_s3_aligned[i]
            short_signal = short_breakout and price_below_ema and volume_spike[i] and not_ranging
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches Camarilla S3 OR trend turns bearish (price below EMA) OR chop increases
            if (close[i] < camarilla_s3_aligned[i] or not price_above_ema or chop[i] > 65):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Camarilla R3 OR trend turns bullish (price above EMA) OR chop increases
            if (close[i] > camarilla_r3_aligned[i] or not price_below_ema or chop[i] > 65):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_AvoidChop"
timeframe = "4h"
leverage = 1.0