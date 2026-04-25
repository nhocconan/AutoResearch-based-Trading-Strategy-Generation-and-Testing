#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hVolumeSpike_TrendFilter
Hypothesis: 4h Camarilla R1/S1 breakout with 12h volume spike (>2.0x 20-bar mean) and 12h EMA50 trend filter (price > EMA50 for long, < EMA50 for short). Uses HTF 12h for volume and trend alignment. Targets 20-30 trades/year per symbol by requiring strong volume spike and clear trend. Designed to work in both bull (breakouts with volume) and bear (trend-following shorts) markets via disciplined entry/exit. Focus on BTC/ETH as primary symbols.
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
    
    # Get 12h data for HTF volume and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA(50) on 12h for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h ATR(14) for Camarilla levels (using 12h HLC)
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - pd.Series(close_12h).shift(1)))
    tr3 = pd.Series(np.abs(low_12h - pd.Series(close_12h).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla R1/S1 levels from previous 12h bar
    # R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12 (standard Camarilla)
    camarilla_r1 = close_12h + 1.1 * (high_12h - low_12h) / 12
    camarilla_s1 = close_12h - 1.1 * (high_12h - low_12h) / 12
    
    # Align Camarilla levels to 4h timeframe (use previous bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Volume confirmation: current 12h volume > 2.0x 20-bar mean volume
    vol_mean_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (vol_mean_20_12h * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in uptrend (price > EMA50) with volume confirmation
            # Short: price breaks below Camarilla S1 in downtrend (price < EMA50) with volume confirmation
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema_50_12h_aligned[i]) and (vol_spike_aligned[i] > 0.5)
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema_50_12h_aligned[i]) and (vol_spike_aligned[i] > 0.5)
            
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
            # Exit when price moves back below EMA50 (trend reversal)
            exit_signal = close[i] < ema_50_12h_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above EMA50 (trend reversal)
            exit_signal = close[i] > ema_50_12h_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hVolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0