#!/usr/bin/env python3
# Strategy: 6h_12h_Pivot_R2S2_Breakout_Volume_ATRFilter_v1
# Hypothesis: Breakout above 12h pivot R2 or below S2 with volume confirmation and ATR-based volatility filter.
# Uses 6h candles for entries, filtered by 12h pivot levels and volume > 2x 20-period MA.
# ATR filter ensures we only trade when volatility is sufficient to avoid whipsaws in low-volatility periods.
# Designed for 15-35 trades/year to minimize fee drag and work in both bull and bear markets.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for pivot levels and ATR
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h ATR(14) for volatility filter
    high_low = high_12h - low_12h
    high_close = np.abs(high_12h - np.roll(close_12h, 1))
    low_close = np.abs(low_12h - np.roll(close_12h, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h Pivot points (R2, S2)
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    R2 = pivot_12h + (range_12h * 1.1 / 6)  # R2 = pivot + 1.1 * range / 6
    S2 = pivot_12h - (range_12h * 1.1 / 6)  # S2 = pivot - 1.1 * range / 6
    
    # Align 12h indicators to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    R2_aligned = align_htf_to_ltf(prices, df_12h, R2)
    S2_aligned = align_htf_to_ltf(prices, df_12h, S2)
    
    # Load 6h data for entry timing and volume
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume spike detection (20-period on 6h)
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_14_aligned[i]) or np.isnan(R2_aligned[i]) or 
            np.isnan(S2_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        atr = atr_14_aligned[i]
        
        # ATR filter: only trade when ATR is above its 50-period MA (avoid low volatility)
        atr_ma_50 = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).mean()[i]
        if np.isnan(atr_ma_50) or atr < atr_ma_50:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R2, with volume confirmation
            if price > R2_aligned[i] and vol > 2.0 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2, with volume confirmation
            elif price < S2_aligned[i] and vol > 2.0 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S2
            if price < S2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R2
            if price > R2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_Pivot_R2S2_Breakout_Volume_ATRFilter_v1"
timeframe = "6h"
leverage = 1.0