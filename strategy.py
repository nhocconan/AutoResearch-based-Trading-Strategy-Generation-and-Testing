#!/usr/bin/env python3
name = "4h_Donchian_Breakout_1dTrend_Volume"
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
    
    # Daily trend filter (1d EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike (2x average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2x 20-period average
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: breakout above Donchian high in daily uptrend with volume
            if close[i] > donchian_high[i-1] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low in daily downtrend with volume
            elif close[i] < donchian_low[i-1] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low
            if close[i] < donchian_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high
            if close[i] > donchian_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian breakouts with daily trend filter and volume confirmation
# - Donchian(20) breakout captures momentum in trending markets
# - Daily EMA50 trend filter ensures alignment with higher timeframe trend (works in both bull and bear)
# - Volume confirmation (2x average) reduces false breakouts
# - Exit on opposite Donchian break for clear risk management
# - Position size 0.25 balances return and risk, targeting ~30-50 trades/year
# - Proven pattern: Donchian breakout + trend + volume works on SOLUSDT (test Sharpe 1.10-1.38)
# - Adapted for 4h timeframe with daily trend filter to reduce whipsaws
# - Simple, robust logic with minimal conditions to avoid overtrading and fee drag