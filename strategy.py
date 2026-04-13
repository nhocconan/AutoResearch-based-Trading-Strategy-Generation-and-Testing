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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period Donchian channels on 1d
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period SMA on 1d for trend filter
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(sma_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to SMA50
        above_sma = close[i] > sma_50_aligned[i]
        below_sma = close[i] < sma_50_aligned[i]
        
        # Volatility filter: avoid low volatility environments
        volatility_filter = atr_14_aligned[i] > 0
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_high_20_aligned[i]
        breakout_down = close[i] < donch_low_20_aligned[i]
        
        # Entry conditions with trend alignment
        long_entry = breakout_up and above_sma and volatility_filter
        short_entry = breakout_down and below_sma and volatility_filter
        
        # Exit conditions: opposite breakout or volatility collapse
        exit_long = position == 1 and (breakout_down or atr_14_aligned[i] < atr_14_aligned[i-1] * 0.5)
        exit_short = position == -1 and (breakout_up or atr_14_aligned[i] < atr_14_aligned[i-1] * 0.5)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_sma50_atr_filter"
timeframe = "4h"
leverage = 1.0