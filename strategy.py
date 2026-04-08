#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_v4
# Hypothesis: Trade Donchian(20) breakouts on 4h with 1d trend filter and volume confirmation.
# Long: 4h high breaks above 20-bar high + 1d close > 1d SMA50 + volume surge.
# Short: 4h low breaks below 20-bar low + 1d close < 1d SMA50 + volume surge.
# Uses ATR(14) stoploss. Designed to capture trends in both bull and bear markets.
# Target: 20-50 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d trend filter: SMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma50_1d_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Donchian break down OR stoploss hit
            if low[i] < low_min_20[i] or close[i] < high_max_20[i] - 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian break up OR stoploss hit
            if high[i] > high_max_20[i] or close[i] > low_min_20[i] + 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Donchian break up + 1d uptrend + volume surge
            if high[i] > high_max_20[i] and close[i] > sma50_1d_aligned[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: Donchian break down + 1d downtrend + volume surge
            elif low[i] < low_min_20[i] and close[i] < sma50_1d_aligned[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals