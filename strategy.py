#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_Regime
Hypothesis: Camarilla R1/S1 breakouts on 4h with 12h EMA50 trend filter and volume spike confirmation.
Only trade breakouts in direction of 12h trend. Uses discrete position sizing (0.25) to minimize fee churn.
Designed for low trade frequency (~20-40/year) to work in both bull and bear markets via trend alignment.
Camarilla levels provide high-probability reversal/breakout points, volume confirms institutional interest,
and 12h trend filter avoids counter-trend whipsaws. Added choppiness regime filter to avoid ranging markets.
Target: 75-150 total trades over 4 years.
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
    
    # Get 12h and 1d data for HTF filters
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from 1d OHLC (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_width = 1.1 * (high_1d - low_1d) / 12.0
    r1_1d = close_1d + camarilla_width
    s1_1d = close_1d - camarilla_width
    
    # Align HTF indicators to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume spike: volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Choppiness regime filter: avoid ranging markets (CHOP > 61.8)
    # CHOP = 100 * log10(sum(ATR,14) / (max(high,14)-min(low,14))) / log10(14)
    atr_14 = pd.Series(np.maximum.reduce([
        high[1:] - close[:-1],
        high[:-1] - close[1:],
        np.abs(high[1:] - low[:-1])
    ])).rolling(14, min_periods=14).mean().values
    atr_14 = np.concatenate([np.full(14, np.nan), atr_14])  # align length
    
    max_high_14 = pd.Series(high).rolling(14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(14, min_periods=14).min().values
    
    chop = 100 * np.log10(pd.Series(atr_14).rolling(14, min_periods=14).sum().values / 
                          np.maximum(max_high_14 - min_low_14, 1e-10)) / np.log10(14)
    chop[np.isnan(chop)] = 100  # default to choppy when not enough data
    chop_regime = chop < 61.8  # trending when CHOP < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50), volume MA (20), CHOP (14+14)
    start_idx = max(50, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for Camarilla breakout signals with all filters
            # Long: price breaks above R1 in uptrend (close > EMA50) + volume spike + trending regime
            # Short: price breaks below S1 in downtrend (close < EMA50) + volume spike + trending regime
            long_signal = (close[i] > r1_aligned[i]) and (close[i] > ema50_aligned[i]) and volume_spike[i] and chop_regime[i]
            short_signal = (close[i] < s1_aligned[i]) and (close[i] < ema50_aligned[i]) and volume_spike[i] and chop_regime[i]
            
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
            # Exit when price moves back below R1 (failed breakout) or trend reverses
            exit_signal = (close[i] < r1_aligned[i]) or (close[i] < ema50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above S1 (failed breakdown) or trend reverses
            exit_signal = (close[i] > s1_aligned[i]) or (close[i] > ema50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0