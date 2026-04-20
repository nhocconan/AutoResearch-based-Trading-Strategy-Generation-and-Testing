#!/usr/bin/env python3
# Strategy: 12h_1d_Camarilla_R1S1_Breakout_Volume_RSIFilter_v1
# Hypothesis: Breakout above daily Camarilla R1 or below S1 with volume confirmation and daily RSI filter on 12h timeframe.
# Uses 12h bars for entries, filtering by daily RSI to avoid overbought/oversold extremes.
# Volume > 2x 20-period MA confirms institutional interest. Designed for 12-37 trades/year to minimize fee drag.
# Works in both bull and bear markets by using RSI filter to avoid extremes and volume confirmation for validity.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF analysis (RSI, pivot levels)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d RSI(14) for filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = np.where(np.isfinite(rsi_14), rsi_14, 100)  # Replace inf/-inf with 100
    rsi_14 = np.nan_to_num(rsi_14, nan=100.0)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Daily Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1 = pivot_1d + (range_1d * 1.1 / 12)
    S1 = pivot_1d - (range_1d * 1.1 / 12)
    
    # Align 1d indicators to 12h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Load 12h data for entry timing, volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume spike detection (20-period on 12h)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        
        if position == 0:
            # Long: price breaks above R1, RSI not overbought (<70), with volume confirmation
            if (price > R1_aligned[i] and 
                rsi_14_aligned[i] < 70 and 
                vol > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, RSI not oversold (>30), with volume confirmation
            elif (price < S1_aligned[i] and 
                  rsi_14_aligned[i] > 30 and 
                  vol > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1
            if price < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1
            if price > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_RSIFilter_v1"
timeframe = "12h"
leverage = 1.0