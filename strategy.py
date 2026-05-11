#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Trend_Filter_1dVWAP
Hypothesis: Donchian(20) breakouts aligned with daily VWAP trend and volume confirmation.
Enters long when price breaks above upper band in uptrend, short when breaks below lower band in downtrend.
Uses daily VWAP as trend filter to avoid counter-trend trades. Volume spike confirms breakout strength.
Designed for low trade frequency (<25/year) to minimize fee drag. Works in bull/bear markets by following higher timeframe trend.
"""

name = "4h_Donchian_Breakout_Trend_Filter_1dVWAP"
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
    
    # === 1d Data for VWAP Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily VWAP calculation
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_num = (typical_price_1d * df_1d['volume'].values).cumsum()
    vwap_den = df_1d['volume'].values.cumsum()
    vwap_1d = vwap_num / vwap_den
    vwap_1d = np.where(vwap_den == 0, np.nan, vwap_1d)
    
    # Align daily VWAP to 4h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === Donchian Channel (20-period) on 4h ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Filter: 1.8x 20-period EMA on 4h ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Donchian and VWAP)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band with uptrend (price > VWAP) and volume spike
            if (close[i] > high_20[i] and 
                close[i] > vwap_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with downtrend (price < VWAP) and volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < vwap_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below lower Donchian band (reversal signal)
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above upper Donchian band (reversal signal)
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals