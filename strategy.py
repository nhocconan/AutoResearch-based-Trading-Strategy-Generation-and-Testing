#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeFilter
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x median). 
In choppy markets (CHOP > 50), fade extreme levels (R3/S3) for mean reversion. 
In trending markets (CHOP <= 50), breakout continuation at R4/S4 levels. 
Uses discrete position sizing (0.25) to minimize churn. Target: 50-150 trades over 4 years.
Works in both bull and bear markets by adapting to regime: mean reversion in chop, trend following in trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 6h (based on previous 6h bar)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    h_6h = df_6h['high'].values
    l_6h = df_6h['low'].values
    c_6h = df_6h['close'].values
    
    # Previous bar's values for level calculation (to avoid look-ahead)
    h_6h_prev = np.roll(h_6h, 1)
    l_6h_prev = np.roll(l_6h, 1)
    c_6h_prev = np.roll(c_6h, 1)
    h_6h_prev[0] = np.nan
    l_6h_prev[0] = np.nan
    c_6h_prev[0] = np.nan
    
    # Calculate Camarilla levels
    rng_6h = h_6h_prev - l_6h_prev
    r3_6h = c_6h_prev + (rng_6h * 1.1 / 4)
    s3_6h = c_6h_prev - (rng_6h * 1.1 / 4)
    r4_6h = c_6h_prev + (rng_6h * 1.1 / 2)
    s4_6h = c_6h_prev - (rng_6h * 1.1 / 2)
    
    # Align to 6h primary timeframe
    r3_6h_aligned = align_htf_to_ltf(prices, df_6h, r3_6h)
    s3_6h_aligned = align_htf_to_ltf(prices, df_6h, s3_6h)
    r4_6h_aligned = align_htf_to_ltf(prices, df_6h, r4_6h)
    s4_6h_aligned = align_htf_to_ltf(prices, df_6h, s4_6h)
    
    # Volume confirmation: volume > 1.5x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    # Load 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Choppiness regime filter (14-period) from 1d data
    if len(df_1d) < 20:
        chop_aligned = np.full(n, 50.0)  # neutral default
    else:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        atr_14 = []
        tr = np.maximum(high_1d[1:] - low_1d[1:], np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1])))
        tr = np.concatenate([[np.nan], tr])
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        chop = 100 * np.log10(atr_14 * 14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period volume median, 34-period EMA, 14-period chop)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_6h_aligned[i]) or np.isnan(s3_6h_aligned[i]) or 
            np.isnan(r4_6h_aligned[i]) or np.isnan(s4_6h_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Regime detection: CHOP > 50 = ranging/mean reverting, CHOP <= 50 = trending
        in_choppy_regime = chop_aligned[i] > 50
        
        if in_choppy_regime:
            # Choppy market: mean reversion at extreme levels (R3/S3)
            # Long: price breaks below S3 (oversold) with volume confirmation
            if close[i] < s3_6h_aligned[i] and volume_confirm[i]:
                if position != 1:
                    signals[i] = base_size
                    position = 1
                else:
                    signals[i] = base_size
            # Short: price breaks above R3 (overbought) with volume confirmation
            elif close[i] > r3_6h_aligned[i] and volume_confirm[i]:
                if position != -1:
                    signals[i] = -base_size
                    position = -1
                else:
                    signals[i] = -base_size
            # Exit: price returns to midpoint (mean reversion complete)
            elif position == 1 and close[i] > (r3_6h_aligned[i] + s3_6h_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] < (r3_6h_aligned[i] + s3_6h_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = base_size
                else:
                    signals[i] = -base_size
        else:
            # Trending market: breakout continuation at R4/S4 with trend filter
            # Long: price breaks above R4 + volume confirmation + bullish 1d trend
            if close[i] > r4_6h_aligned[i] and volume_confirm[i] and close[i] > ema_34_1d_aligned[i]:
                if position != 1:
                    signals[i] = base_size
                    position = 1
                else:
                    signals[i] = base_size
            # Short: price breaks below S4 + volume confirmation + bearish 1d trend
            elif close[i] < s4_6h_aligned[i] and volume_confirm[i] and close[i] < ema_34_1d_aligned[i]:
                if position != -1:
                    signals[i] = -base_size
                    position = -1
                else:
                    signals[i] = -base_size
            # Exit: opposite breakout (price returns to the other extreme level)
            elif position == 1 and close[i] < s3_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > r3_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = base_size
                else:
                    signals[i] = -base_size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0