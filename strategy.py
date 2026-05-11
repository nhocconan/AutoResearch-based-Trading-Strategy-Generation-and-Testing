#!/usr/bin/env python3
"""
6h_Keltner_MeanReversion_1dTrend
Hypothesis: Trade mean reversion at Keltner lower/upper bands with 1d trend filter and volume confirmation. 
In trending markets, price pulls back to the 20 EMA (Keltner middle) before continuing. 
In ranging markets, price reverts from the bands. Volume confirms momentum exhaustion.
Targets 15-25 trades/year on 6h to minimize fee drag while capturing mean reversion edges.
"""

name = "6h_Keltner_MeanReversion_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h data for Keltner Bands (20 EMA, ATR(10)*2) ===
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    # Calculate EMA20 and ATR(10) on 1h
    ema20_1h = pd.Series(df_1h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr_1h = np.maximum(
        df_1h['high'].values - df_1h['low'].values,
        np.maximum(
            np.abs(df_1h['high'].values - np.concatenate([[df_1h['close'][0]], df_1h['close'][:-1]])),
            np.abs(df_1h['low'].values - np.concatenate([[df_1h['close'][0]], df_1h['close'][:-1]]))
        )
    )
    atr10_1h = pd.Series(tr_1h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Bands: middle = EMA20, upper/lower = EMA20 ± 2*ATR
    keltner_middle_1h = ema20_1h
    keltner_upper_1h = ema20_1h + 2 * atr10_1h
    keltner_lower_1h = ema20_1h - 2 * atr10_1h
    
    # Align 1h Keltner bands to 6h
    km_6h = align_htf_to_ltf(prices, df_1h, keltner_middle_1h)
    ku_6h = align_htf_to_ltf(prices, df_1h, keltner_upper_1h)
    kl_6h = align_htf_to_ltf(prices, df_1h, keltner_lower_1h)
    
    # === Daily Trend Filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Filter (1.3x 20-period EMA on 6h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 1h and daily calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(km_6h[i]) or np.isnan(ku_6h[i]) or np.isnan(kl_6h[i]) or 
            np.isnan(ema50_6h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches or crosses below lower band with uptrend and volume
            if (close[i] <= kl_6h[i] and 
                close[i] > ema50_6h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches or crosses above upper band with downtrend and volume
            elif (close[i] >= ku_6h[i] and 
                  close[i] < ema50_6h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle band (mean reversion complete) or trend breaks
            if close[i] >= km_6h[i] or close[i] < ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price returns to middle band or trend breaks
            if close[i] <= km_6h[i] or close[i] >= ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals