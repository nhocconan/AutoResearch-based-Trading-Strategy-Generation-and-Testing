#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian Channel Breakout with 1-day ATR filter and volume confirmation.
Trades breakouts above/below 20-period Donchian channels only when ATR volatility is elevated
(confirmed breakout) and volume exceeds 1.5x 20-period average. Uses 1-day ATR to filter
low-volatility environments where breakouts fail. Designed for low trade frequency (20-40 trades/year)
to minimize fee drift and capture strong directional moves in both bull and bear markets.
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
    
    # Load daily data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 4-hour Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility and volume filters
        vol_filter = atr_14_aligned[i] > np.nanmedian(atr_14_aligned[max(0, i-50):i+1])
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and vol_filter and vol_spike:
            # Long: breakout above Donchian high
            if high[i] > donchian_high[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low
            elif low[i] < donchian_low[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite Donchian level retest or volatility collapse
            exit_signal = False
            
            if position == 1:
                # Exit long: price retests Donchian low or volatility drops
                if low[i] < donchian_low[i] or atr_14_aligned[i] < 0.8 * np.nanmedian(atr_14_aligned[max(0, i-50):i+1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price retests Donchian high or volatility drops
                if high[i] > donchian_high[i] or atr_14_aligned[i] < 0.8 * np.nanmedian(atr_14_aligned[max(0, i-50):i+1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dATR_Volume"
timeframe = "4h"
leverage = 1.0