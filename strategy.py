#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R1/S1 breakout with 1d ATR volume confirmation and 12h EMA50 trend filter.
Long when price breaks above Camarilla R1 level AND 12h EMA50 uptrend AND volume > 2.5x ATR-based threshold.
Short when price breaks below Camarilla S1 level AND 12h EMA50 downtrend AND volume > 2.5x ATR-based threshold.
Exit when price crosses 12h EMA50 or Camarilla Pivot point.
Designed for 6h timeframe to achieve 12-37 trades/year with discrete sizing (0.25) to minimize fee churn.
Uses ATR-based volume threshold to adapt to volatility regimes, working in both bull and bear markets.
"""

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
    
    # Load 1d data for Camarilla levels and ATR - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volume threshold
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    range_1d = high_1d - low_1d
    camarilla_r1_1d = close_1d + 1.125 * range_1d   # R1: close + 1.125 * range
    camarilla_s1_1d = close_1d - 1.125 * range_1d   # S1: close - 1.125 * range
    camarilla_pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Ensure warmup for EMA50 and vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr14_1d_aligned[i]
        
        # Dynamic volume threshold: 2.5x ATR-based volume average
        vol_threshold = 2.5 * atr_val * vol_ma_val if atr_val > 0 else 2.0 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND 12h EMA50 uptrend AND volume spike
            if (price > camarilla_r1_aligned[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and  # EMA50 rising
                volume[i] > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 AND 12h EMA50 downtrend AND volume spike
            elif (price < camarilla_s1_aligned[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and  # EMA50 falling
                  volume[i] > vol_threshold):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses 12h EMA50 (trend reversal)
            if position == 1 and price < ema50_12h_aligned[i]:
                exit_signal = True
            elif position == -1 and price > ema50_12h_aligned[i]:
                exit_signal = True
            # Secondary exit: price crosses Camarilla Pivot (mean reversion)
            elif position == 1 and price < camarilla_pivot_aligned[i]:
                exit_signal = True
            elif position == -1 and price > camarilla_pivot_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R1S1_1dATR_Volume_12hEMA50_Trend"
timeframe = "6h"
leverage = 1.0