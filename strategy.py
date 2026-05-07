#!/usr/bin/env python3
name = "12h_Donchian_20_WeeklyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h and 1w data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_12h) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donchian_high = np.full_like(high_12h, np.nan)
    donchian_low = np.full_like(low_12h, np.nan)
    
    for i in range(20, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-20:i])
        donchian_low[i] = np.min(low_12h[i-20:i])
    
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 12h volume filter: > 2x 20-period average
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 2.0 * vol_ma_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Wait for EMA50 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_12h_aligned[i]) or np.isnan(donchian_low_12h_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume filter and weekly uptrend
            if (close[i] > donchian_high_12h_aligned[i] and vol_filter[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume filter and weekly downtrend
            elif (close[i] < donchian_low_12h_aligned[i] and vol_filter[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below Donchian low or trend reversal
            if close[i] < donchian_low_12h_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above Donchian high or trend reversal
            if close[i] > donchian_high_12h_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian breakouts on 12h timeframe with weekly trend filter and volume confirmation
# capture strong trends while avoiding false breakouts. Weekly EMA50 ensures we only trade
# in the direction of the higher timeframe trend. Volume filter ensures breakouts have
# conviction. Position size 0.25 limits risk. Target 15-25 trades/year to minimize fee drag.