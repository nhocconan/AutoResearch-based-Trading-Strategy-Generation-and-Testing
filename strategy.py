#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume spike confirmation.
- Uses Donchian(20) breakout from 4h price structure for entry signals
- 12h EMA(50) as trend filter (long only above, short only below)
- Volume > 1.8x 20-period average for confirmation
- Position size: 0.30 discrete level to balance return and fee drag
- Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted breakouts
"""

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
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian(20) from 4h data (upper = 20-period high, lower = 20-period low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(50)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i]  # Close above upper band
        breakout_down = close[i] < donchian_low[i]  # Close below lower band
        
        if position == 0:
            # Long: Donchian breakout up AND price above 12h EMA50 AND volume confirmation
            if breakout_up and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short: Donchian breakout down AND price below 12h EMA50 AND volume confirmation
            elif breakout_down and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown OR price crosses below 12h EMA50
            if breakout_down or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Donchian breakout OR price crosses above 12h EMA50
            if breakout_up or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_VolumeSpike_Filter_v1"
timeframe = "4h"
leverage = 1.0