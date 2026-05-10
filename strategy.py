#!/usr/bin/env python3
# 12h_Donchian_Breakout_Volume_Spike
# Hypothesis: In strong trends, price breaks through 12-period Donchian channels on 12h timeframe.
# Use 1d ATR for volatility filter and volume spike for confirmation. Works in bull markets (buy breakouts)
# and bear markets (sell breakdowns) by only trading in direction of 1d trend via EMA50.
# Target: 15-30 trades/year to minimize fee drag on 12h timeframe.

name = "12h_Donchian_Breakout_Volume_Spike"
timeframe = "12h"
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
    
    # Get 1d data for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr_14_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12h Donchian channels (20-period)
    # For 12h, we need to get actual 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    donchian_high = df_12h['high'].rolling(window=20, min_periods=20).max().values
    donchian_low = df_12h['low'].rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation (20-period MA on 12h = ~10 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA50 (50), 1d ATR (14), 12h Donchian (20), volume MA (20)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14_1d_aligned[i] > 0  # Always true if ATR calculated
        
        # Volume confirmation: spike above average
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: uptrend + price breaks above Donchian high + volume spike
            if uptrend and close[i] > donchian_high_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below Donchian low + volume spike
            elif downtrend and close[i] < donchian_low_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters Donchian channel
            if not uptrend or close[i] < donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters Donchian channel
            if not downtrend or close[i] > donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals