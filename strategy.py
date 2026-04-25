#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_RegimeFilter
Hypothesis: Trade 4h Camarilla H3/L3 breakouts with 1d ATR-based volatility filter (ATR > 1.5x 20-day MA) to capture high-momentum moves, volume confirmation (>2.0x 20-bar 4h MA), and choppiness regime filter (CHOP < 61.8 for trending markets). 
This strategy targets volatile, trending markets to capture strong directional moves while avoiding low-volatility false breakouts. Uses discrete sizing 0.25 to balance profit and fee drag. Target: 20-35 trades/year (~80-140 over 4 years) to stay within fee drag limits.
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
    
    # Get 1d data for ATR-based volatility filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    volatility_filter = atr_14_1d_aligned > (1.5 * atr_ma_20_1d_aligned)
    
    # Calculate Camarilla levels from previous 1d bar's OHLC
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high_1d - prev_low_1d
    h3 = prev_close_1d + 1.1 * camarilla_range / 6   # H3 level
    l3 = prev_close_1d - 1.1 * camarilla_range / 6   # L3 level
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current 4h volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Choppiness regime filter: CHOP < 61.8 indicates trending market (use 14-period)
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr1_4h[0] = 0
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    abs_changes = np.abs(np.diff(close, prepend=close[0]))
    sum_abs_changes = pd.Series(abs_changes).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_abs_changes / (atr_14_4h * 14)) / np.log10(10)
    chop_regime = chop < 61.8  # Trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d ATR (14+20=34), volume MA (20), and CHOP (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(volatility_filter[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND volatility filter AND volume confirm AND trending regime
            long_setup = (close[i] > h3_aligned[i]) and \
                         volatility_filter[i] and \
                         volume_confirm[i] and \
                         chop_regime[i]
            # Short: price breaks below L3 AND volatility filter AND volume confirm AND trending regime
            short_setup = (close[i] < l3_aligned[i]) and \
                          volatility_filter[i] and \
                          volume_confirm[i] and \
                          chop_regime[i]
            
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
            # Exit: price re-enters Camarilla H3/L3 range OR volatility filter turns off OR chop regime turns ranging
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (not volatility_filter[i]) or \
               (not chop_regime[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3/L3 range OR volatility filter turns off OR chop regime turns ranging
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (not volatility_filter[i]) or \
               (not chop_regime[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0