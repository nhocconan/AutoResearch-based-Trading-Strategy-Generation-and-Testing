#!/usr/bin/env python3
name = "4h_Donchian20_Breakout_12hTrend_Volume"
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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA(20) for trend filter
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Donchian(20) channels on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 3-period average (3 bars = 6h)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 3)  # Wait for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_3[i] * 2.0
            uptrend = ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1]
            
            if close[i] > highest_high[i-1] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band with volume and 12h downtrend
            elif close[i] < lowest_low[i-1] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below lower band or volume drops
            if close[i] < lowest_low[i-1] or volume[i] < vol_ma_3[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above upper band or volume drops
            if close[i] > highest_high[i-1] or volume[i] < vol_ma_3[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian(20) breakout with 12h trend and volume confirmation
# - Donchian(20) breakout captures momentum with clear entry/exit levels
# - 12h EMA(20) trend filter ensures alignment with higher timeframe momentum
# - Volume spike (2.0x average) confirms institutional participation
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price returns to opposite Donchian band or volume weakens
# - Position size 0.25 targets ~25-40 trades/year, avoiding fee drag
# - Uses 12h trend to avoid whipsaws in ranging markets
# - Volume confirmation reduces false breakouts
# - Simple 3-condition logic minimizes overfitting and parameter sensitivity
# - Designed for BTC/ETH primary focus with proven Donchian effectiveness