#!/usr/bin/env python3
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian Channel (20)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1d ATR (14) for volatility filter
    tr1 = pd.Series(df_1d['high'] - df_1d['low'])
    tr2 = pd.Series(abs(df_1d['high'] - df_1d['close'].shift(1)))
    tr3 = pd.Series(abs(df_1d['low'] - df_1d['close'].shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 1d Volume MA (20) for volume confirmation
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: current 15m volatility vs 1d ATR
        # Approximate 15m volatility as 1/4 of daily (since 1d = 96*15m)
        vol_filter = atr_14_aligned[i] > 0
        
        # Volume confirmation: current volume > 1.5x 1d average volume
        vol_confirm = vol > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian high + volume confirmation
            if price > donchian_high_aligned[i] and vol_confirm:
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Donchian low + volume confirmation
            elif price < donchian_low_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian low
            if price < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian high
            if price > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dDonchian20_Volume_Breakout_v1"
timeframe = "12h"
leverage = 1.0