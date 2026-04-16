#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions on 6h timeframe.
# 1d EMA34 acts as trend filter: only long when price > EMA34, short when price < EMA34.
# Volume confirmation: current 6h volume > 1.5x 20-period average of 6h volume.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in both bull and bear markets via trend filter and mean-reversion entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data for Williams %R, volume, ATR ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === Williams %R (14-period) ===
    highest_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_14 - close_6h) / (highest_14 - lowest_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # === 6h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    # === 1d EMA34 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        wr = williams_r_aligned[i]
        ema34_val = ema34_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume
        
        # === EXIT LOGIC (Williams %R mean reversion) ===
        if position == 1:  # Long position
            # Exit when Williams %R rises above -20 (overbought)
            if wr > -20:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -80 (oversold)
            if wr < -80:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: Williams %R < -80 (oversold) AND price > EMA34 AND volume confirmation
            if wr < -80 and price > ema34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: Williams %R > -20 (overbought) AND price < EMA34 AND volume confirmation
            elif wr > -20 and price < ema34_val and vol_confirm:
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

name = "6h_WilliamsR_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0