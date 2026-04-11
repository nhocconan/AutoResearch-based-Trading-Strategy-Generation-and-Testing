#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ATR MA (20-period) for volatility regime filter
    atr_ma_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR ratio (current / MA) for regime detection
    atr_ratio_1d = np.where(atr_ma_20_1d > 0, atr_14_1d / atr_ma_20_1d, 1.0)
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 1d price range (high - low) for breakout levels
    range_1d = high_1d - low_1d
    
    # Align 1d range to 4h timeframe
    range_1d_aligned = align_htf_to_ltf(prices, df_1d, range_1d)
    
    # Calculate 4h Donchian channel (20-period) for breakout confirmation
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average on 4h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(range_1d_aligned[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volatility filter: trade only when volatility is elevated (ATR ratio > 1.2)
        vol_filter = atr_ratio_1d_aligned[i] > 1.2
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Breakout conditions using 1d range
        breakout_level = range_1d_aligned[i]  # Use 1d range as breakout threshold
        
        # Long: Price breaks above 4h high + volatility + volume
        enter_long = (price_high > high_20[i]) and vol_filter and vol_confirm
        
        # Short: Price breaks below 4h low + volatility + volume
        enter_short = (price_low < low_20[i]) and vol_filter and vol_confirm
        
        # Exit conditions: reverse signal or volatility contraction
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on short signal or volatility contraction
            exit_long = enter_short or (atr_ratio_1d_aligned[i] < 0.8)
        elif position == -1:
            # Exit short on long signal or volatility contraction
            exit_short = enter_long or (atr_ratio_1d_aligned[i] < 0.8)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals