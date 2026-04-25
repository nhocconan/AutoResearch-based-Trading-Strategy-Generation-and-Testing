#!/usr/bin/env python3
"""
6h_PivotRange_Reversion_1dVolSpike_ATRFilter
Hypothesis: Trade reversals from 1d Camarilla H3/L3 levels on 6h timeframe with volume spike and ATR filter.
In ranging markets (BTC/ETH 2025+), price tends to revert from extreme daily levels.
Volume spike confirms institutional interest at these levels.
ATR filter avoids whipsaws in low volatility.
Discrete sizing 0.25 balances profit and fee drag. Target: 12-25 trades/year (~50-100 over 4 years).
Works in bull/bear: mean reversion works in ranging markets, volatility filter adapts to conditions.
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
    
    # Get 1d data for Camarilla levels and volume filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Camarilla levels from previous 1d bar's OHLC
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high_1d - prev_low_1d
    h3 = prev_close_1d + 1.1 * camarilla_range / 6   # H3 level
    l3 = prev_close_1d - 1.1 * camarilla_range / 6   # L3 level
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current 1d volume > 2.0x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume_1d > (2.0 * vol_ma_20_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d ATR (14) and volume MA (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below L3 AND volume spike AND ATR > 0 (avoid dead markets)
            long_setup = (close[i] < l3_aligned[i]) and \
                         volume_spike_aligned[i] and \
                         (atr_14_1d_aligned[i] > 0)
            # Short: price above H3 AND volume spike AND ATR > 0
            short_setup = (close[i] > h3_aligned[i]) and \
                          volume_spike_aligned[i] and \
                          (atr_14_1d_aligned[i] > 0)
            
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
            # Exit: price crosses above L3 (mean reversion) OR volume spike ends
            if (close[i] > l3_aligned[i]) or \
               (~volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price crosses below H3 (mean reversion) OR volume spike ends
            if (close[i] < h3_aligned[i]) or \
               (~volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_PivotRange_Reversion_1dVolSpike_ATRFilter"
timeframe = "6h"
leverage = 1.0