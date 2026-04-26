#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_RegimeFilter_v1
Hypothesis: On 12h timeframe, trade Camarilla R1/S1 breakouts with 1-week EMA50 trend filter and choppiness regime (CHOP < 61.8 = trending). Uses volume spike (>1.5x median) for confirmation. Designed for low trade frequency (12-37/year) to minimize fee drag while capturing strong directional moves in both bull and bear markets via HTF trend alignment and regime filter.
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
    
    # Get 1w data for HTF trend (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior 1d)
    cam_high = pd.Series(df_1d['high'].values).shift(1).values
    cam_low = pd.Series(df_1d['low'].values).shift(1).values
    cam_close = pd.Series(df_1d['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels (core breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # Get 1d data for choppiness regime (CHOP(14))
    chop_length = 14
    if len(df_1d) < chop_length * 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr1).ewm(span=chop_length, adjust=False, min_periods=chop_length).mean().values
    
    # Highest high and lowest low over chop_length periods
    hh = pd.Series(high_1d).rolling(window=chop_length, min_periods=chop_length).max().values
    ll = pd.Series(low_1d).rolling(window=chop_length, min_periods=chop_length).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(atr1) / (hh - ll)) / log10(chop_length)
    sum_atr = pd.Series(atr_1d).rolling(window=chop_length, min_periods=chop_length).sum().values
    denominator = hh - ll
    denominator = np.where(denominator == 0, 1e-10, denominator)  # avoid division by zero
    chop = 100 * np.log10(sum_atr / denominator) / np.log10(chop_length)
    
    # Volume spike filter: volume > 1.5x median volume (30-period) for high conviction
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # ATR(12) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Align HTF indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(50) 1w, Camarilla (need 2 bars for shift), chop (28), volume median (30), ATR (12)
    start_idx = max(50, 2, 28, 30, 12) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        chop_val = chop_aligned[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1w_val
        downtrend = close_val < ema_50_1w_val
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_regime = chop_val < 61.8
        
        # Volume spike filter: only trade in high-volume environments
        volume_spike = volume_val > 1.5 * vol_median_val
        
        if position == 0:
            # Long: break above R1 with volume spike, uptrend, and trending regime
            long_signal = (close_val > r1_val) and \
                          volume_spike and \
                          uptrend and \
                          trending_regime
            
            # Short: break below S1 with volume spike, downtrend, and trending regime
            short_signal = (close_val < s1_val) and \
                           volume_spike and \
                           downtrend and \
                           trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop (2.0x ATR)
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop (2.0x ATR)
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0