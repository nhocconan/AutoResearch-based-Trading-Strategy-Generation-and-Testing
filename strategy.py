#!/usr/bin/env python3
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
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility filter (use previous day's ATR)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily ATR-based channels for mean reversion signals
    atr_mult = 1.5
    upper_channel = close_1d + atr_1d * atr_mult
    lower_channel = close_1d - atr_1d * atr_mult
    
    # Align to 12h timeframe
    upper_channel_12h = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_12h = align_htf_to_ltf(prices, df_1d, lower_channel)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike detection (using 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period - ensure we have enough data for all indicators
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel_12h[i]) or np.isnan(lower_channel_12h[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Mean reversion long: price touches lower channel with volume spike
            if (close[i] <= lower_channel_12h[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Mean reversion short: price touches upper channel with volume spike
            elif (close[i] >= upper_channel_12h[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to mean or volatility drops
            if (close[i] >= (close[i-1]) or volume_spike[i] == False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to mean or volatility drops
            if (close[i] <= (close[i-1]) or volume_spike[i] == False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ATR_MeanReversion_VolumeSpike"
timeframe = "12h"
leverage = 1.0