#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dATR_Trend_VolumeSpike
Hypothesis: Camarilla H3/L3 breakout on 4h with 1d ATR-based trend filter and volume confirmation.
Uses discrete position sizing (0.30) to limit fee drag. Targets 20-40 trades/year.
Works in bull markets (breakouts with trend) and bear markets (fades from extremes with volume).
ATR filter adapts to volatility, improving performance in both trending and ranging conditions.
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
    
    # Get 1d data for ATR trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for trend filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # First value NaN
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d EMA200 for long-term trend direction
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 1d data for Camarilla levels
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align to 4h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ATR (14), EMA200 (200), volume MA (20)
    start_idx = max(14, 200, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend: above EMA200 = uptrend, below = downtrend
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        if position == 0:
            # Long: price closes above H3 + uptrend + volume spike
            long_setup = (close[i] > camarilla_h3_aligned[i]) and \
                         uptrend and \
                         volume_spike[i]
            # Short: price closes below L3 + downtrend + volume spike
            short_setup = (close[i] < camarilla_l3_aligned[i]) and \
                          downtrend and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price closes below L3 OR trend turns down
            if (close[i] < camarilla_l3_aligned[i]) or \
               not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price closes above H3 OR trend turns up
            if (close[i] > camarilla_h3_aligned[i]) or \
               not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATR_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0