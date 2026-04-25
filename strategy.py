#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dATR_Trend_VolumeSpike
Hypothesis: Trade 4h Camarilla R3/S3 breakouts (wider bands) with 1d ATR trend filter (price > 1d ATR-based mean) and volume spike (>2.0x 20-bar MA). Wider breakout bands reduce false signals, ATR trend filter adapts to volatility, volume confirmation ensures institutional participation. Discrete sizing 0.25 targets 20-50 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and ATR trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_range = (high_1d - low_1d) * 1.1 / 4.0
    camarilla_R3 = close_1d + camarilla_range
    camarilla_S3 = close_1d - camarilla_range
    
    # Align Camarilla levels to 4h (completed 1d bar only)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Calculate ATR on 1d for trend filter (ATR(14))
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align length
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d ATR-based trend: price > close + 0.5*ATR (bullish) or price < close - 0.5*ATR (bearish)
    atr_trend_bull = close_1d + (0.5 * atr_14_1d)
    atr_trend_bear = close_1d - (0.5 * atr_14_1d)
    
    # Align ATR trend levels to 4h
    atr_trend_bull_aligned = align_htf_to_ltf(prices, df_1d, atr_trend_bull)
    atr_trend_bear_aligned = align_htf_to_ltf(prices, df_1d, atr_trend_bear)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Camarilla (1d), ATR (14), volume MA (20)
    start_idx = max(20, 14)  # 20 for volume MA, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(atr_trend_bull_aligned[i]) or 
            np.isnan(atr_trend_bear_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + above ATR bull trend + volume spike
            long_setup = (close[i] > camarilla_R3_aligned[i]) and \
                         (close[i] > atr_trend_bull_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below Camarilla S3 + below ATR bear trend + volume spike
            short_setup = (close[i] < camarilla_S3_aligned[i]) and \
                          (close[i] < atr_trend_bear_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below Camarilla S3 OR below ATR bear trend
            if (close[i] < camarilla_S3_aligned[i]) or \
               (close[i] < atr_trend_bear_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Camarilla R3 OR above ATR bull trend
            if (close[i] > camarilla_R3_aligned[i]) or \
               (close[i] > atr_trend_bull_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dATR_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0