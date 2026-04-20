#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility regime
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
    
    # Calculate daily range for breakout levels
    range_1d = high_1d - low_1d
    range_ma_10 = pd.Series(range_1d).rolling(window=10, min_periods=10).mean().values
    range_ma_10_aligned = align_htf_to_ltf(prices, df_1d, range_ma_10)
    
    # Calculate daily close momentum
    mom_1d = close_1d - np.roll(close_1d, 1)
    mom_ma_5 = pd.Series(mom_1d).rolling(window=5, min_periods=5).mean().values
    mom_ma_5_aligned = align_htf_to_ltf(prices, df_1d, mom_ma_5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        atr_val = atr_14_aligned[i]
        range_ma_val = range_ma_10_aligned[i]
        mom_val = mom_ma_5_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(atr_val) or np.isnan(range_ma_val) or 
            np.isnan(mom_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: positive momentum + low volatility (range < ATR) + price > open
            if mom_val > 0 and range_ma_val < atr_val * 0.5 and close_val > prices['open'].iloc[i]:
                signals[i] = 0.25
                position = 1
            # Short: negative momentum + low volatility + price < open
            elif mom_val < 0 and range_ma_val < atr_val * 0.5 and close_val < prices['open'].iloc[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: momentum turns negative or volatility increases
            if mom_val < 0 or range_ma_val > atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: momentum turns positive or volatility increases
            if mom_val > 0 or range_ma_val > atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_VolMomentum_ATRRegime
# Uses 1-day momentum and volatility regime for 12h entries
# Enters long when momentum positive, low volatility (range < 0.5*ATR), and bullish candle
# Enters short when momentum negative, low volatility, and bearish candle
# Exits when momentum reverses or volatility increases (range > ATR)
# Designed for 12h timeframe with ~15-25 trades/year
name = "12h_VolMomentum_ATRRegime"
timeframe = "12h"
leverage = 1.0