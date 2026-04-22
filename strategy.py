#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian(40) breakout with 1-week EMA(34) trend filter and volume spike confirmation.
Trades breakouts in the direction of the weekly trend only when volume exceeds 1.8x the 20-period average.
Uses fixed position sizing (0.25) to limit risk and reduce trade frequency.
Targets 15-30 trades/year (60-120 total over 4 years) to minimize fee drag and improve generalization.
Works in both bull and bear markets by aligning with higher timeframe trend and requiring strong volume confirmation.
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
    
    # Load 4h data for Donchian channel - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    # Calculate 4h Donchian Channel (40-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=40, min_periods=40).max().values
    donchian_low = pd.Series(low_4h).rolling(window=40, min_periods=40).min().values
    
    # Align Donchian levels
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # 1w EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above Donchian high, above 1w EMA (uptrend)
            if close[i] > donchian_high_aligned[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below 1w EMA (downtrend)
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches Donchian low or closes below 1w EMA
                if close[i] < donchian_low_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches Donchian high or closes above 1w EMA
                if close[i] > donchian_high_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_40_1wEMA34_Volume"
timeframe = "4h"
leverage = 1.0