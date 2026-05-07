#!/usr/bin/env python3
name = "1d_Donchian20_1wTrend_Volume_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 5-period average (5-day average)
    vol_ma_5 = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian and weekly EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(vol_ma_5[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_5[i] * 2.0
            weekly_uptrend = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]
            
            if close[i] > donch_high[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low with volume and weekly downtrend
            elif close[i] < donch_low[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or volume drops
            if close[i] < donch_low[i] or volume[i] < vol_ma_5[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or volume drops
            if close[i] > donch_high[i] or volume[i] < vol_ma_5[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# - Donchian(20) breakout captures momentum in trending markets
# - Weekly EMA(20) filter ensures alignment with higher timeframe trend
# - Volume spike (2.0x 5-day average) confirms institutional participation
# - Works in both bull (buy breakouts in weekly uptrend) and bear (sell breakdowns in weekly downtrend)
# - Exit on reversal or volume weakening to avoid whipsaws
# - Position size 0.25 targets ~20-50 trades/year to minimize fee drag
# - Uses actual weekly EMA from 1w data, not daily approximation
# - Designed to avoid overtrading with strict volume and trend conditions
# - Weekly trend filter reduces false signals during counter-trend moves