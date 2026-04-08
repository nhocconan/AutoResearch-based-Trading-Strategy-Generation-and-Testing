#!/usr/bin/env python3
# 4h_1d_donchian_volume_rsi_v1
# Hypothesis: 4-hour Donchian breakout with 1-day volume confirmation and RSI filter to avoid false breakouts.
# Long when: price breaks above 4h Donchian high (20), 1d volume > 20-period SMA, and RSI(14) > 50.
# Short when: price breaks below 4h Donchian low (20), 1d volume > 20-period SMA, and RSI(14) < 50.
# Exit when price reverses to Donchian midpoint or volume filter fails.
# Uses 4h for price channel, 1d for volume filter, and 4h RSI for entry quality.
# Target: 20-40 trades/year to minimize fee drag while capturing strong momentum.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_rsi_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[14] = np.mean(gain[:14])
    avg_loss[14] = np.mean(loss[:14])
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h Donchian channels (20)
    donch_high = np.zeros(n)
    donch_low = np.zeros(n)
    donch_high[:] = np.nan
    donch_low[:] = np.nan
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Calculate 4h Donchian midpoint for exit
    donch_mid = (donch_high + donch_low) / 2
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_sma_20 = np.zeros(len(vol_1d))
    vol_sma_20[:] = np.nan
    vol_sma_20[19] = np.mean(vol_1d[:20])
    for i in range(20, len(vol_1d)):
        vol_sma_20[i] = (vol_sma_20[i-1] * 19 + vol_1d[i]) / 20
    
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data is not ready
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(rsi[i]) or np.isnan(vol_sma_20_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_sma_20_aligned[i] if vol_sma_20_aligned[i] > 0 else 0
        
        if position == 1:  # Long
            # Exit: price below Donchian midpoint OR volume < average
            if close[i] < donch_mid[i] or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price above Donchian midpoint OR volume < average
            if close[i] > donch_mid[i] or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry: Donchian breakout with volume confirmation and RSI filter
            if close[i] > donch_high[i] and vol_ratio > 1.5 and rsi[i] > 50:
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and vol_ratio > 1.5 and rsi[i] < 50:
                position = -1
                signals[i] = -0.25
    
    return signals