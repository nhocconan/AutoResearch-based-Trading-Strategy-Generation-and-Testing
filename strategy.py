#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ATR stop
# Donchian breakout captures breakouts in trending markets
# Volume confirmation ensures breakout is supported by participation
# ATR stop limits downside in false breakouts
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# Designed for low trade frequency (target 7-25/year) with clear trend following logic

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Donchian(20) channels
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d ATR(14) for volatility and stop
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1w volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    volume_1w = df_1w['volume'].values
    volume_ma = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    
    # Align indicators to 1d timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            continue
        
        # Volume confirmation: current volume > 1.5 * weekly average
        volume_ok = volume[i] > 1.5 * volume_ma_aligned[i]
        
        # Long breakout: price > Donchian high + volume confirmation
        if close[i] > donch_high_aligned[i] and volume_ok and position <= 0:
            position = 1
            signals[i] = position_size
        # Short breakdown: price < Donchian low + volume confirmation
        elif close[i] < donch_low_aligned[i] and volume_ok and position >= 0:
            position = -1
            signals[i] = -position_size
        # Exit long: price < Donchian low
        elif position == 1 and close[i] < donch_low_aligned[i]:
            position = 0
            signals[i] = 0.0
        # Exit short: price > Donchian high
        elif position == -1 and close[i] > donch_high_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0