#!/usr/bin/env python3
name = "12h_Donchian20_Trix12_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_high_12h = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_12h = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # 1w TRIX (12-period EMA triple)
    close_1w = df_1w['close'].values
    ema1 = pd.Series(close_1w).ewm(span=12, adjust=False).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False).mean()
    trix = ((ema3 - pd.Series(ema3).shift(1)) / pd.Series(ema3).shift(1)) * 100
    trix_values = trix.values
    trix_12h = align_htf_to_ltf(prices, df_1w, trix_values)
    
    # Volume confirmation (12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or np.isnan(trix_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + TRIX > 0 + volume confirmation
            if (close[i] > donch_high_12h[i]) and (trix_12h[i] > 0) and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + TRIX < 0 + volume confirmation
            elif (close[i] < donch_low_12h[i]) and (trix_12h[i] < 0) and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below Donchian low
            if close[i] < donch_low_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above Donchian high
            if close[i] > donch_high_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals