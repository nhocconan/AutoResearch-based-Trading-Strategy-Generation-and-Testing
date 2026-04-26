#!/usr/bin/env python3
"""
4h_CamR1S1_Donchian20_VolumeRegime_12hTrend
Hypothesis: Combine Camarilla R1/S1 breakout with Donchian(20) channel confirmation, 
volume spike filter, and 12h EMA50 trend filter. Enter long when price breaks above 
Camarilla R1 AND Donchian upper band with volume > 2x MA and 12h uptrend. Enter short 
when price breaks below Camarilla S1 AND Donchian lower band with volume > 2x MA and 
12h downtrend. Uses discrete position sizing (0.25) to minimize fee churn. 
Designed for 4h timeframe to capture institutional moves while avoiding overtrading.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
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
    
    # Calculate Camarilla levels from previous day
    # Need daily OHLC from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC (aligned to 4h bars)
    prev_close_1d = df_1d['close'].shift(1).values  # Previous day close
    prev_high_1d = df_1d['high'].shift(1).values    # Previous day high
    prev_low_1d = df_1d['low'].shift(1).values      # Previous day low
    
    # Align to 4h timeframe
    prev_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Calculate Camarilla levels
    R1 = prev_close_1d_aligned + (prev_high_1d_aligned - prev_low_1d_aligned) * 1.1 / 12
    S1 = prev_close_1d_aligned - (prev_high_1d_aligned - prev_low_1d_aligned) * 1.1 / 12
    
    # Calculate Donchian(20) channels
    lookback = 20
    upper_chan = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_chan = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 2x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    uptrend_12h = close > ema_50_12h_aligned
    downtrend_12h = close < ema_50_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian/volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(upper_chan[i]) or 
            np.isnan(lower_chan[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price > R1 AND price > upper Donchian band + volume + 12h uptrend
            if close[i] > R1[i] and close[i] > upper_chan[i] and volume_confirm[i] and uptrend_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 AND price < lower Donchian band + volume + 12h downtrend
            elif close[i] < S1[i] and close[i] < lower_chan[i] and volume_confirm[i] and downtrend_12h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price < S1 OR 12h trend changes to downtrend
            if close[i] < S1[i] or not uptrend_12h[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price > R1 OR 12h trend changes to uptrend
            if close[i] > R1[i] or not downtrend_12h[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_CamR1S1_Donchian20_VolumeRegime_12hTrend"
timeframe = "4h"
leverage = 1.0