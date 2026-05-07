# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "4h_Donchian20_VolumeTrend_1dTrend"
timeframe = "4h"
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
    
    # Load daily data ONCE before loop for trend filter and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily Donchian(20) for breakout signals
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 6-period average (1 day of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above Donchian high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_6[i] * 1.5
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] > high_20_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below Donchian low with volume and daily downtrend
            elif close[i] < low_20_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian high or trend reverses
            if close[i] < high_20_aligned[i] or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian low or trend reverses
            if close[i] > low_20_aligned[i] or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and daily trend filter
# - Donchian(20) breakout captures trend continuation moves
# - Volume spike (1.5x average) confirms institutional participation
# - Daily EMA(50) trend filter ensures trades align with higher timeframe momentum
# - Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
# - Exit when price retests breakout level or trend reverses
# - Position size 0.25 targets ~25-50 trades/year, avoiding fee drag
# - Uses daily Donchian levels for better stability vs intraday noise
# - Volume confirmation reduces false breakouts in choppy markets
# - Simple 3-condition logic minimizes overfitting and curve fitting
# - Designed for BTC/ETH primary focus with applicability to SOL