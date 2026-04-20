#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate daily Williams %R (14) for mean reversion
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate daily ATR (14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_21_1w_aligned[i]
        wr_val = williams_r_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(wr_val) or np.isnan(atr_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with price above weekly EMA
            if wr_val < -80 and close_val > ema_val and atr_1d_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with price below weekly EMA
            elif wr_val > -20 and close_val < ema_val and atr_1d_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R overbought (> -20) or price below weekly EMA
            if wr_val > -20 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R oversold (< -80) or price above weekly EMA
            if wr_val < -80 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_WilliamsR_EMA21Trend
# Uses 1-week EMA(21) for trend filter and daily Williams %R(14) for mean reversion signals
# Enters long when Williams %R < -80 (oversold) and price above weekly EMA
# Enters short when Williams %R > -20 (overbought) and price below weekly EMA
# Exits on opposite Williams %R level or price crossing weekly EMA
# Designed for 6h timeframe with ~10-30 trades/year
name = "6h_WilliamsR_EMA21Trend"
timeframe = "6h"
leverage = 1.0