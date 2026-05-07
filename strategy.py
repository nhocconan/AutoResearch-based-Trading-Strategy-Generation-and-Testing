#!/usr/bin/env python3
name = "12h_Donchian20_Trend_Filter"
timeframe = "12h"
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
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 25 or len(df_1d) < 25:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (already 12h data)
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h volume spike: > 2.0x 20-period average
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume > 2.0 * vol_ma_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 50)  # Wait for Donchian and EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_12h_aligned[i]) or np.isnan(donchian_low_12h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume spike and uptrend (price > EMA50)
            if (close[i] > donchian_high_12h_aligned[i] and vol_spike_12h[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: Break below Donchian low with volume spike and downtrend (price < EMA50)
            elif (close[i] < donchian_low_12h_aligned[i] and vol_spike_12h[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: Price below Donchian low or trend reversal (price < EMA50)
            if close[i] < donchian_low_12h_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Price above Donchian high or trend reversal (price > EMA50)
            if close[i] > donchian_high_12h_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Donchian breakouts on 12h with volume confirmation and 1d EMA50 trend filter capture
# strong momentum moves in both bull and bear markets. Volume spike ensures institutional participation.
# Trend filter avoids counter-trend trades. Position size 0.30 limits risk. Target ~25 trades/year.
# Exit on retrace to opposite Donchian band or trend reversal. Simple, robust, low-frequency.