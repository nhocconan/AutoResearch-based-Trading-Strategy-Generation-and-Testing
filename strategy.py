#!/usr/bin/env python3
# Strategy: 12h_1d_Pivot_R1S1_Breakout_VolumeATRFilter
# Hypothesis: Price breakouts above daily R1 or below daily S1 with volume confirmation and ATR-based trend filter capture strong momentum moves. The ATR filter ensures trades align with volatility regime, reducing false breakouts in low-volatility environments. Works in bull/bear markets by capturing breakout momentum regardless of direction. Targets 15-30 trades/year by requiring clear breakouts with volume surge and ATR filter.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF analysis
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for trend filter (14-period on 1d)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily pivot points
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1 = pivot_1d + (range_1d * 1.1 / 12)
    S1 = pivot_1d - (range_1d * 1.1 / 12)
    
    # Align pivot levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 12h data for entry timing and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume spike detection (20-period)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # ATR for volatility filter (14-period on 12h)
    high_low_12h = high_12h - low_12h
    high_close_12h = np.abs(high_12h - np.roll(close_12h, 1))
    low_close_12h = np.abs(low_12h - np.roll(close_12h, 1))
    high_low_12h[0] = high_12h[0] - low_12h[0]
    high_close_12h[0] = np.abs(high_12h[0] - close_12h[0])
    low_close_12h[0] = np.abs(low_12h[0] - close_12h[0])
    tr_12h = np.maximum(high_low_12h, np.maximum(high_close_12h, low_close_12h))
    tr_12h[0] = high_low_12h[0]
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_14_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and ATR filter (avoid low volatility breakouts)
            if (price > R1_aligned[i] and 
                vol > 2.0 * vol_ma_20_aligned[i] and
                atr_14_12h_aligned[i] > 0.5 * atr_14_1d_aligned[i]):  # Ensure sufficient volatility
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation and ATR filter
            elif (price < S1_aligned[i] and 
                  vol > 2.0 * vol_ma_20_aligned[i] and
                  atr_14_12h_aligned[i] > 0.5 * atr_14_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below pivot or ATR-based stop
            if (price < pivot_1d_aligned[i] or 
                price < low_12h[i] - 2.0 * atr_14_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above pivot or ATR-based stop
            if (price > pivot_1d_aligned[i] or 
                price > high_12h[i] + 2.0 * atr_14_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_R1S1_Breakout_VolumeATRFilter"
timeframe = "12h"
leverage = 1.0