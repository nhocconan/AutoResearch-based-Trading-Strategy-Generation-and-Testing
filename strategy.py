#!/usr/bin/env python3
# 12h_1d_1w_volume_breakout_momentum_v1
# Hypothesis: 12h price breaks Donchian(20) channel with 1d volume confirmation and 1w momentum filter.
# Long when price breaks above Donchian high + volume spike + 1w bullish momentum.
# Short when price breaks below Donchian low + volume spike + 1w bearish momentum.
# Uses volume spike (>1.5x 20-period average) and 1w RSI(14) > 50 for long, < 50 for short.
# Designed for 15-30 trades/year on 12h to avoid fee drag. Works in bull/bear via momentum filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_volume_breakout_momentum_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i+1])
        donchian_low[i] = np.min(low[i-20:i+1])
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i+1])
    volume_spike = volume > (vol_ma * 1.5)
    
    # Get 1d data for volume confirmation (optional secondary confirmation)
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i+1])
    volume_spike_1d = vol_1d > (vol_ma_1d * 1.5)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get 1w data for momentum filter (RSI)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # RSI(14) on 1w
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1w), np.nan)
    avg_loss = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):
        if i == 14:
            avg_gain[i] = np.mean(gain[i-14:i+1])
            avg_loss[i] = np.mean(loss[i-14:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(rsi_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or momentum turns bearish
            if close[i] < donchian_low[i] or rsi_1w_aligned[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or momentum turns bullish
            if close[i] > donchian_high[i] or rsi_1w_aligned[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high + volume spike + bullish momentum
            if (close[i] > donchian_high[i] and 
                (volume_spike[i] or volume_spike_1d_aligned[i]) and 
                rsi_1w_aligned[i] > 50):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + volume spike + bearish momentum
            elif (close[i] < donchian_low[i] and 
                  (volume_spike[i] or volume_spike_1d_aligned[i]) and 
                  rsi_1w_aligned[i] < 50):
                position = -1
                signals[i] = -0.25
    
    return signals