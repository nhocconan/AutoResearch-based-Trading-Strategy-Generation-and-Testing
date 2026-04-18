#!/usr/bin/env python3
"""
4h_KAMA_Direction_1dATRBreakout_VolumeSpike_v1
Hypothesis: KAMA trend direction on 4h combined with 1-day ATR-based breakouts and volume spikes captures momentum while filtering chop. KAMA adapts to market conditions, reducing whipsaw in ranging markets. ATR breakouts ensure volatility expansion accompanies moves. Volume confirms institutional participation. Designed for 20-35 trades/year to minimize fee drag.
"""

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
    
    # KAMA on 4h for trend direction
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change.rolling(window=10, min_periods=10).sum() / volatility.replace(0, np.nan)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1-day ATR for breakout levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]
    atr_1d = np.zeros_like(close_1d)
    atr_1d[0] = tr1[0]
    for i in range(1, len(tr1)):
        atr_1d[i] = 0.9 * atr_1d[i-1] + 0.1 * tr1[i]
    atr_1d_smooth = pd.Series(atr_1d).rolling(window=5, min_periods=5).mean().values
    
    # Calculate breakout levels: today's open ± 1.5 * ATR(1d)
    open_1d = df_1d['open'].values
    breakout_up = open_1d + 1.5 * atr_1d_smooth
    breakout_down = open_1d - 1.5 * atr_1d_smooth
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down)
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(breakout_up_aligned[i]) or
            np.isnan(breakout_down_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        bu = breakout_up_aligned[i]
        bd = breakout_down_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above breakout_up with KAMA up and volume spike
            if price > bu and price > kama_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below breakout_down with KAMA down and volume spike
            elif price < bd and price < kama_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to breakout_down or KAMA turns down
            if price < bd or price < kama_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to breakout_up or KAMA turns up
            if price > bu or price > kama_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Direction_1dATRBreakout_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0