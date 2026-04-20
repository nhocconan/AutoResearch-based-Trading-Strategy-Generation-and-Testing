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
    
    # Calculate weekly 200-period SMA for long-term trend (slow, stable)
    close_1w = df_1w['close'].values
    sma_200_1w = pd.Series(close_1w).rolling(window=200, min_periods=200).mean().values
    sma_200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_200_1w)
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate daily volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        sma_val = sma_200_1w_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol_val = volume_1d[i]
        vol_avg_val = vol_avg_20_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(sma_val) or np.isnan(rsi_val) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly SMA200 (uptrend), RSI < 30 (oversold), volume above average
            if close_val > sma_val and rsi_val < 30 and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly SMA200 (downtrend), RSI > 70 (overbought), volume above average
            elif close_val < sma_val and rsi_val > 70 and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly SMA200 or RSI > 70 (overbought)
            if close_val < sma_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly SMA200 or RSI < 30 (oversold)
            if close_val > sma_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_1wSMA200_1dRSI_Volume
# Uses 1-week 200-period SMA for long-term trend direction
# Enters long when price above weekly SMA200, RSI < 30, and volume above average
# Enters short when price below weekly SMA200, RSI > 70, and volume above average
# Exits when price crosses weekly SMA200 or RSI reaches opposite extreme
# Designed for 12h timeframe with ~12-37 trades/year (50-150 total over 4 years)
name = "12h_1wSMA200_1dRSI_Volume"
timeframe = "12h"
leverage = 1.0