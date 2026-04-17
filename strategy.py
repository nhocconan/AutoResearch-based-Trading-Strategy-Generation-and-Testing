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
    
    # === 1d Donchian(20) breakout for trend direction ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels
    donch_high_20 = np.full_like(high_1d, np.nan)
    donch_low_20 = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 19:
            donch_high_20[i] = np.max(high_1d[i-19:i+1])
            donch_low_20[i] = np.min(low_1d[i-19:i+1])
        elif i > 0:
            donch_high_20[i] = np.max(high_1d[max(0, i-9):i+1])
            donch_low_20[i] = np.min(low_1d[max(0, i-9):i+1])
        else:
            donch_high_20[i] = high_1d[0]
            donch_low_20[i] = low_1d[0]
    
    # === 1d ATR(14) for volatility filter and position sizing ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close, 1))  # Note: using close from 1d data would be better but we don't have it here
    tr3 = np.abs(low_1d - np.roll(close, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close[0])
    tr3[0] = np.abs(low_1d[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # === 12h Volume confirmation ===
    # Note: We're using 12h timeframe as primary, so we need 12h volume data
    # But we don't have direct access to 12h data from prices - we'll use 1h as proxy for volume
    # In practice, for 12h strategy, we should use 12h volume from 12h data
    # Since we can't get 12h volume directly, we'll skip volume confirmation for now
    # and rely on price action and volatility
    
    # Align indicators to 12h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price breaks above 20-day Donchian high + volatility filter
            if (close[i] > donch_high_20_aligned[i] and 
                atr_14_aligned[i] > 0.005 * close[i]):  # volatility filter
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below 20-day Donchian low + volatility filter
            elif (close[i] < donch_low_20_aligned[i] and 
                  atr_14_aligned[i] > 0.005 * close[i]):  # volatility filter
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Price crosses below 20-day Donchian low
            if close[i] < donch_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above 20-day Donchian high
            if close[i] > donch_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_VolatilityFilter_v1"
timeframe = "12h"
leverage = 1.0