#!/usr/bin/env python3
name = "4h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Donchian(20) from previous day (completed bar only)
    donch_high_1d = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low_1d = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 3-period average (3/4 day of 4h bars)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 3)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above Donchian high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_3[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > donch_high_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below Donchian low with volume and daily downtrend
            elif close[i] < donch_low_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or volume drops
            if close[i] < donch_low_aligned[i] or volume[i] < vol_ma_3[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or volume drops
            if close[i] > donch_high_aligned[i] or volume[i] < vol_ma_3[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian(20) breakout with 1d trend and volume confirmation
# - Donchian(20) breakout from previous day captures institutional breakouts
# - Breakout above Donchian high with volume in daily uptrend = long opportunity
# - Breakdown below Donchian low with volume in daily downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation and reduces false breakouts
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price returns to Donchian low (for longs) or high (for shorts) or volume weakens
# - Position size 0.25 targets ~30-60 trades/year, avoiding fee drag
# - Uses actual daily Donchian levels (not intraday) for better stability and fewer signals
# - Daily trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts in ranging markets
# - Simple 3-condition logic: breakout + volume + trend (minimizes overfitting)