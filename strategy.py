#!/usr/bin/env python3
"""
1h_VolumeTrend_Confirmation
Hypothesis: In 1h timeframe, use 4h trend direction (EMA20) as filter and enter on volume spikes with price momentum.
Works in bull (long when 4h EMA up + volume spike + close > open) and bear (short when 4h EMA down + volume spike + close < open).
Volume confirms institutional interest, reducing false signals. Low trade frequency via strict 4h trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    # 1h price momentum: close > open for bull, close < open for bear
    bull_momentum = close > prices['open'].values
    bear_momentum = close < prices['open'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for 4h EMA20 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h uptrend + volume spike + bullish momentum
            if ema_4h_aligned[i] > ema_4h_aligned[i-1] and vol_spike[i] and bull_momentum[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + volume spike + bearish momentum
            elif ema_4h_aligned[i] < ema_4h_aligned[i-1] and vol_spike[i] and bear_momentum[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h trend turns down OR volume spike ends
            if ema_4h_aligned[i] < ema_4h_aligned[i-1] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h trend turns up OR volume spike ends
            if ema_4h_aligned[i] > ema_4h_aligned[i-1] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeTrend_Confirmation"
timeframe = "1h"
leverage = 1.0