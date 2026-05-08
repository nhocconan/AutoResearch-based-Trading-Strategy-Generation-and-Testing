#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d RSI filter and volume confirmation.
# Long when price breaks above upper BB(20,2) and 1d RSI > 55 and volume > 1.5x average.
# Short when price breaks below lower BB(20,2) and 1d RSI < 45 and volume > 1.5x average.
# Exit when price crosses the 20-period SMA or RSI reverts to neutral zone (45-55).
# Uses 1d RSI as a higher timeframe filter to avoid counter-trend trades.
# Volume confirmation ensures breakouts have conviction.
# Designed to work in both bull (breakouts continue) and bear (mean reversion in range) markets.
# Target: 20-50 trades per year to minimize fee drag.

name = "4h_BB_Breakout_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[:14])
    avg_loss[14] = np.mean(loss[:14])
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.concatenate([[np.nan], rsi_1d])
    
    # Align 1d RSI to 4h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(vol_ma[i]) or
            np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper BB, RSI > 55, volume > 1.5x average
            if (close[i] > upper_bb[i] and 
                rsi_1d_aligned[i] > 55 and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price < lower BB, RSI < 45, volume > 1.5x average
            elif (close[i] < lower_bb[i] and 
                  rsi_1d_aligned[i] < 45 and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below SMA or RSI returns to neutral
            if close[i] < sma_20[i] or (rsi_1d_aligned[i] >= 45 and rsi_1d_aligned[i] <= 55):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above SMA or RSI returns to neutral
            if close[i] > sma_20[i] or (rsi_1d_aligned[i] >= 45 and rsi_1d_aligned[i] <= 55):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals