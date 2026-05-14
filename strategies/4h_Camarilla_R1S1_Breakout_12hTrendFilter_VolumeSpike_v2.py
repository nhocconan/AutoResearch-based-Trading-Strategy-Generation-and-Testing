#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrendFilter_VolumeSpike_v2
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume confirmation.
Uses ATR-based volume spike filter (volume > 1.5 * ATR) and discrete sizing (0.25).
Reduced trade frequency by tightening trend filter and adding minimum holding period.
Target: 20-30 trades/year to minimize fee drag while maintaining edge in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    range_1d = df_1d['high'].values - df_1d['low'].values
    
    # Camarilla R1 and S1 levels (main breakout levels)
    camarilla_r1 = typical_price_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = typical_price_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate ATR for volume spike filter (adaptive to volatility)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period
    
    # Start index: need warmup for 12h EMA50 (50) and ATR (14)
    start_idx = max(50, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume spike: current volume > 1.5 * ATR (adaptive threshold)
        volume_spike = volume[i] > 1.5 * atr[i]
        
        if position == 0:
            # Long: price breaks above R1 AND 12h trend bullish (close > EMA50) AND volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and \
                         (close[i] > ema_50_12h_aligned[i]) and \
                         volume_spike
            # Short: price breaks below S1 AND 12h trend bearish (close < EMA50) AND volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and \
                          (close[i] < ema_50_12h_aligned[i]) and \
                          volume_spike
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: price re-enters between S1 and R1 OR 12h trend turns bearish OR min holding period (6 bars = 1 day)
            if (camarilla_s1_aligned[i] < close[i] < camarilla_r1_aligned[i]) or \
               (close[i] < ema_50_12h_aligned[i]) or \
               (bars_since_entry >= 6):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price re-enters between S1 and R1 OR 12h trend turns bullish OR min holding period (6 bars = 1 day)
            if (camarilla_s1_aligned[i] < close[i] < camarilla_r1_aligned[i]) or \
               (close[i] > ema_50_12h_aligned[i]) or \
               (bars_since_entry >= 6):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrendFilter_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0