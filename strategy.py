#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Donchian channel and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Donchian channel (20-day high/low)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max()
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min()
    
    # ATR (14-day)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Weekly EMA (50-period) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Align 1d data to 4h
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20.values)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20.values)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14.values)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only long when price > weekly EMA50, short when price < weekly EMA50
        long_trend = close[i] > ema_50_1w_aligned[i]
        short_trend = close[i] < ema_50_1w_aligned[i]
        
        # Breakout conditions
        long_breakout = high[i] > donchian_high_20_aligned[i]
        short_breakout = low[i] < donchian_low_20_aligned[i]
        
        # Volatility filter: require sufficient ATR to avoid choppy markets
        atr_ratio = atr_14_aligned[i] / (donchian_high_20_aligned[i] - donchian_low_20_aligned[i] + 1e-10)
        volatility_filter = atr_ratio > 0.01  # Minimum volatility threshold
        
        # Entry conditions
        if position == 0:
            if long_breakout and long_trend and volatility_filter:
                position = 1
                signals[i] = position_size
            elif short_breakout and short_trend and volatility_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Stop loss: 2 * ATR below entry (approximated by Donchian low)
            if low[i] < donchian_low_20_aligned[i] - (2 * atr_14_aligned[i]):
                position = 0
                signals[i] = 0.0
            # Take profit: 3 * ATR above Donchian low (trailing)
            elif high[i] > donchian_low_20_aligned[i] + (3 * atr_14_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Stop loss: 2 * ATR above entry (approximated by Donchian high)
            if high[i] > donchian_high_20_aligned[i] + (2 * atr_14_aligned[i]):
                position = 0
                signals[i] = 0.0
            # Take profit: 3 * ATR below Donchian high (trailing)
            elif low[i] < donchian_high_20_aligned[i] - (3 * atr_14_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Breakout_ATR_Trend"
timeframe = "4h"
leverage = 1.0