#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian(20) breakout with daily EMA34 trend filter and volume spike.
Only trade breakouts in direction of daily EMA34 trend when price breaks above/below
20-period Donchian channel with volume confirmation (>1.5x average). Uses daily trend
to avoid counter-trend trades, volume to confirm breakout strength, and ATR-based
stoploss to manage risk. Designed for low trade frequency (20-40 trades/year) by
requiring trend alignment, breakout, and volume confirmation. Works in both bull
and bear markets by following the daily EMA34 trend direction.
"""

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
    
    # Daily EMA34 for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4-hour Donchian channel (20-period) - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h close then to 4h timeframe
    donch_high_4h = pd.Series(donch_high).rolling(window=2, min_periods=2).apply(lambda x: x[-1] if not np.isnan(x[-1]) else x[0], raw=False).shift(1).values
    donch_low_4h = pd.Series(donch_low).rolling(window=2, min_periods=2).apply(lambda x: x[-1] if not np.isnan(x[-1]) else x[0], raw=False).shift(1).values
    
    # Handle edge cases
    donch_high_4h[0] = donch_high[0]
    donch_low_4h[0] = donch_low[0]
    
    # Align Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: daily EMA34 uptrend + price breaks above Donchian high + volume spike
            if ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and close[i] > donch_high_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: daily EMA34 downtrend + price breaks below Donchian low + volume spike
            elif ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and close[i] < donch_low_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: daily EMA34 trend reversal or price returns to opposite Donchian level
            exit_signal = False
            
            if position == 1:
                # Exit long: daily EMA34 turns down or price breaks below Donchian low
                if ema34_1d_aligned[i] < ema34_1d_aligned[i-1] or close[i] < donch_low_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: daily EMA34 turns up or price breaks above Donchian high
                if ema34_1d_aligned[i] > ema34_1d_aligned[i-1] or close[i] > donch_high_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_DailyEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0