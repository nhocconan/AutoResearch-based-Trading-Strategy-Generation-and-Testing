#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h Williams %R (14) for mean reversion signals
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_12h_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate 1d EMA (34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR (14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
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
        wr_val = williams_r_12h_aligned[i]
        ema_val = ema_34_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(wr_val) or np.isnan(ema_val) or np.isnan(atr_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) and price above 1d EMA(34)
            if wr_val < -80 and close_val > ema_val and atr_1d_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) and price below 1d EMA(34)
            elif wr_val > -20 and close_val < ema_val and atr_1d_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns above -50 or price crosses below EMA
            if wr_val > -50 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns below -50 or price crosses above EMA
            if wr_val < -50 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_WilliamsEMA_MeanReversion_V1
# Uses 6-hour Williams %R (14) for mean reversion signals
# Enters long when Williams %R < -80 (oversold) and price above 1d EMA(34)
# Enters short when Williams %R > -20 (overbought) and price below 1d EMA(34)
# Uses 1d EMA(34) as trend filter to avoid counter-trend trades
# Exits when Williams %R returns to -50 level or price crosses EMA
# Designed for 6h timeframe with ~12-37 trades/year
name = "6h_WilliamsEMA_MeanReversion_V1"
timeframe = "6h"
leverage = 1.0