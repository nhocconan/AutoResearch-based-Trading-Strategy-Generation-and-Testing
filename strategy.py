#!/usr/bin/env python3
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
    
    # === Daily OHLC for ATR and volume calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === ATR(14) for volatility filter ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # === 12-hour price channel (Donchian-like: 2-period high/low) ===
    # For 12h timeframe, use 2-period lookback (previous two 12h candles)
    # We'll approximate using 1d data: 2-period high/low from 1d
    high_2p = pd.Series(high_1d).rolling(window=2, min_periods=2).max().values
    low_2p = pd.Series(low_1d).rolling(window=2, min_periods=2).min().values
    
    # Align HTF data to 12h timeframe
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    volume_spike_12h = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    high_2p_12h = align_htf_to_ltf(prices, df_1d, high_2p)
    low_2p_12h = align_htf_to_ltf(prices, df_1d, low_2p)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_12h[i]) or np.isnan(volume_spike_12h[i]) or
            np.isnan(high_2p_12h[i]) or np.isnan(low_2p_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr = atr_12h[i]
        vol_spike = volume_spike_12h[i]
        high_channel = high_2p_12h[i]
        low_channel = low_2p_12h[i]
        
        # === EXIT LOGIC: Exit when price moves against channel or volatility drops ===
        if position == 1:  # Long position
            # Exit when price drops below lower channel or volatility drops
            if price < low_channel or atr < (atr_12h[i-1] * 0.7 if i > 0 else atr):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above upper channel or volatility drops
            if price > high_channel or atr < (atr_12h[i-1] * 0.7 if i > 0 else atr):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper channel with volume spike
            if price > high_channel and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below lower channel with volume spike
            elif price < low_channel and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_ChannelBreakout_VolumeSpike_ATRFilter"
timeframe = "12h"
leverage = 1.0