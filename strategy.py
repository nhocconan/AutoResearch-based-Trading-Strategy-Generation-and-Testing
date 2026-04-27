#!/usr/bin/env python3
"""
6h_Keltner_Channel_R3_S4_Breakout_1dTrend_Volume
Hypothesis: Price breaking above Keltner upper band (R4) or below lower band (S4) with volume confirmation and 1d trend filter captures strong momentum moves. Works in bull/bear via trend filter.
Target: 15-30 trades/year on 6h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtrf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d trend: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Keltner Channel (20, 2.0)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    upper_band = ema20 + (2.0 * atr)
    lower_band = ema20 - (2.0 * atr)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and ATR
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        upper = upper_band[i]
        lower = lower_band[i]
        
        if position == 0:
            # Long: close breaks above upper band + volume spike + uptrend
            if close[i] > upper and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: close breaks below lower band + volume spike + downtrend
            elif close[i] < lower and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close returns inside Keltner channel or trend turns down
            if close[i] < ema20[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close returns inside Keltner channel or trend turns up
            if close[i] > ema20[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Keltner_Channel_R3_S4_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0