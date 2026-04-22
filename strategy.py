#!/usr/bin/env python3

"""
Hypothesis: 1-day Donchian channel breakout with 1-week EMA trend filter and volume confirmation.
Trades breakouts above/below 20-period Donchian channels in the direction of the 1w EMA trend.
Uses volume spike confirmation to avoid false breakouts. Designed for low trade frequency
(7-25 trades/year) to minimize fee drift and work in both bull and bear markets by aligning
with higher timeframe trend. Focus on BTC/ETH as primary targets.
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
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d Donchian channels (20-period) - calculated on 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
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

name = "1d_Donchian_Breakout_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0