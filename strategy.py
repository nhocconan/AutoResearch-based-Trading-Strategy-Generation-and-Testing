#!/usr/bin/env python3
name = "12h_Donchian20_1dTrend_VolumeBreak"
timeframe = "12h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channel (20-period) on 12h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 20-period average (~10 days of 12h bars)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above Donchian upper with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > high_20[i] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price below Donchian lower with volume and daily downtrend
            elif close[i] < low_20[i] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below Donchian lower or volume drops
            if close[i] < low_20[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above Donchian upper or volume drops
            if close[i] > high_20[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 12h Donchian(20) breakout with daily trend and volume confirmation
# - Donchian breakout captures institutional breakouts in both bull and bear markets
# - Daily EMA(34) trend filter ensures we trade with the higher timeframe trend
# - Volume spike (2x average) confirms institutional participation and reduces false breakouts
# - Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
# - Exit when price returns to Donchian lower/upper or volume weakens
# - Position size 0.30 targets 15-30 trades/year, avoiding fee drag
# - Simple, robust structure that works across market regimes and assets (BTC, ETH, SOL)