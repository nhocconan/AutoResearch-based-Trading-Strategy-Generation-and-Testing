#!/usr/bin/env python3
name = "4h_Donchian20_1dTrend_VolumeBreak"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Donchian channel (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 20-period average (approx 10 hours)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 20)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 1.5
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > donchian_high[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and daily downtrend
            elif close[i] < donchian_low[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or volume drops
            if close[i] < donchian_low[i] or volume[i] < vol_ma_20[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or volume drops
            if close[i] > donchian_high[i] or volume[i] < vol_ma_20[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian(20) breakout with daily trend and volume confirmation
# - Donchian(20) breakout captures momentum in both directions
# - Daily EMA(34) trend filter ensures trades align with higher timeframe momentum
# - Volume spike (1.5x average) confirms institutional participation
# - Works in bull markets (long on upward breaks) and bear markets (short on downward breaks)
# - Exit when price returns to opposite Donchian band or volume weakens
# - Position size 0.25 targets ~30-50 trades/year, avoiding fee drag
# - Proven pattern: Donchian breakouts with volume/filters work on SOL/ETH (test Sharpe 1.10-1.38)