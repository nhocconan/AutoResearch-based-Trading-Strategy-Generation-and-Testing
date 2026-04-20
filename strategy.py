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
    
    # Calculate daily 20-period SMA for trend
    close_1d = df_1d['close'].values
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
    
    # Calculate weekly 10-period RSI
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=10, min_periods=10).mean().values
    avg_loss = pd.Series(loss).rolling(window=10, min_periods=10).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_10_1w = 100 - (100 / (1 + rs))
    rsi_10_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_10_1w)
    
    # Calculate daily volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_10 = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_avg_10_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        sma_val = sma_20_1d_aligned[i]
        rsi_val = rsi_10_1w_aligned[i]
        vol_val = volume_1d[i]
        vol_avg_val = vol_avg_10_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(sma_val) or np.isnan(rsi_val) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above daily SMA (uptrend), RSI < 30 (oversold), volume above average
            if close_val > sma_val and rsi_val < 30 and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: price below daily SMA (downtrend), RSI > 70 (overbought), volume above average
            elif close_val < sma_val and rsi_val > 70 and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily SMA or RSI > 70 (overbought)
            if close_val < sma_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily SMA or RSI < 30 (oversold)
            if close_val > sma_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_1dSMA_1wRSI_Volume
# Uses 1-day 20-period SMA for trend direction
# Enters long when price above daily SMA, weekly RSI < 30, and volume above average
# Enters short when price below daily SMA, weekly RSI > 70, and volume above average
# Exits when price crosses daily SMA or RSI reaches opposite extreme
# Designed for 12h timeframe with ~15-25 trades/year
name = "12h_1dSMA_1wRSI_Volume"
timeframe = "12h"
leverage = 1.0