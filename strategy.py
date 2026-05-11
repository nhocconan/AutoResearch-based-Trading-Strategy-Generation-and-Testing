#!/usr/bin/env python3
"""
4h_Keltner_Breakout_Trend_Volume
Hypothesis: 4h breakouts above/below 2x ATR Keltner channels, filtered by 1d EMA trend and volume spikes.
Trades in direction of 1d trend using previous 1d bar's Keltner levels. Volume confirmation filters false breakouts.
Designed for moderate trade frequency (~50-100/year) to balance opportunity and fee drag. Works in bull/bear by following higher timeframe trend.
"""

name = "4h_Keltner_Breakout_Trend_Volume"
timeframe = "4h"
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
    
    # === 1d Data for Trend Filter and Keltner Channels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Previous 1d bar's OHLC for ATR calculation (Keltner)
    ph_1d = high_1d  # previous 1d high
    pl_1d = low_1d   # previous 1d low
    pc_1d = df_1d['close'].values  # previous 1d close
    
    # ATR(14) on 1d data
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Keltner Channels: 2 * ATR around EMA20
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20_1d + 2 * atr14
    lower_keltner = ema20_1d - 2 * atr14
    
    # Align Keltner levels to 4h
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Filter: 2.0x 20-period EMA on 4h ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 1d EMA34 and ATR)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner with uptrend and volume spike
            if (close[i] > upper_aligned[i] and 
                close[i] > ema34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below lower Keltner with downtrend and volume spike
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price closes below lower Keltner (mean reversion)
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # maintain position
        elif position == -1:
            # Short exit: price closes above upper Keltner (mean reversion)
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # maintain position
    
    return signals