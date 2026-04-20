#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d RSI(14) for overbought/oversold
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d Bollinger Bands (20,2) for mean reversion
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate 4h ATR(14) for stoploss
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_1d_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        atr_val = atr_4h[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(upper_bb_val) or 
            np.isnan(lower_bb_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches lower BB with RSI oversold
            if close_val <= lower_bb_val and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB with RSI overbought
            elif close_val >= upper_bb_val and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches upper BB or ATR stop
            if close_val >= upper_bb_val or close_val < prices['high'].iloc[i] - 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches lower BB or ATR stop
            if close_val <= lower_bb_val or close_val > prices['low'].iloc[i] + 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_BB_RSI_MeanReversion_ATRStop_V1
# Uses daily Bollinger Bands and RSI for mean reversion signals
# Enters long when 4h price touches lower BB with daily RSI < 30
# Enters short when 4h price touches upper BB with daily RSI > 70
# Exits on opposite BB touch or 1.5*ATR stoploss
# Designed for 4h timeframe with ~20-50 trades/year
name = "4h_BB_RSI_MeanReversion_ATRStop_V1"
timeframe = "4h"
leverage = 1.0