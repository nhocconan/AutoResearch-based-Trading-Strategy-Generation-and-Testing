#!/usr/bin/env python3
name = "4h_Donchian20_Breakout_VolumeTrend_12hEMA50"
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
    
    # Load 4h and 12h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_4h) < 20 or len(df_12h) < 10:
        return np.zeros(n)
    
    # 4h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA(50) for trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 4h volume spike: > 2.0x 20-period average (moderate filter)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = volume > 2.0 * vol_ma_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Wait for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume spike, price above 12h EMA50
            if (close[i] > donchian_high[i] and vol_spike_4h[i] and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume spike, price below 12h EMA50
            elif (close[i] < donchian_low[i] and vol_spike_4h[i] and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below Donchian low or trend reversal
            if close[i] < donchian_low[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above Donchian high or trend reversal
            if close[i] > donchian_high[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Donchian breakout with volume confirmation and 12h EMA50 trend filter.
# Position size 0.25 to limit risk. Target ~25-40 trades/year for low fee drag.
# Exits on retrace to opposite Donchian band or trend reversal (price crosses 12h EMA50).
# Works in both bull and bear markets by following the 12h trend direction.