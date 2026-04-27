#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d RSI for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi14_1d = (100 - (100 / (1 + rs))).values
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    
    # 1d ATR for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # 4h Donchian breakout for entry signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # Volume filter: require volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for Donchian and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi14_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + RSI > 50 (bullish) + volume
            if (close[i] > donch_high_20_aligned[i] and 
                rsi14_1d_aligned[i] > 50 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + RSI < 50 (bearish) + volume
            elif (close[i] < donch_low_20_aligned[i] and 
                  rsi14_1d_aligned[i] < 50 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if close[i] < donch_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if close[i] > donch_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_RSI_Volume"
timeframe = "4h"
leverage = 1.0