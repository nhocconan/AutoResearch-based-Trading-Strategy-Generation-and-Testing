#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Confirmation
Hypothesis: Use daily Camarilla pivot levels (R1, S1) for breakout entries on 4h timeframe, confirmed by volume > 1.5x 20-period average and RSI filter to avoid overextended entries. Long when price breaks above R1 with volume confirmation and RSI < 70; short when price breaks below S1 with volume confirmation and RSI > 30. Exit on opposite break or RSI extreme. Targets 20-40 trades/year by requiring multiple confirmations, with position size 0.25. Works in bull/bear via breakout logic and avoids whipsaws with volume/RSI filters.
"""

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
    volume = prices['volume'].values
    
    # Calculate 1-day Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels using previous day's data
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    range_1d = high_1d[:-1] - low_1d[:-1]
    
    R1 = pivot + (range_1d * 1.1 / 12)
    S1 = pivot - (range_1d * 1.1 / 12)
    
    # Align to 4h timeframe (wait for daily bar close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate 20-period volume average
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    # Calculate RSI(14) for overbought/oversold filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need volume avg and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation and RSI not overbought
            if (close[i] > R1_aligned[i] and 
                volume[i] > 1.5 * vol_avg[i] and 
                rsi[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume confirmation and RSI not oversold
            elif (close[i] < S1_aligned[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  rsi[i] > 30):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below S1 or RSI overbought
            if (close[i] < S1_aligned[i] or rsi[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or RSI oversold
            if (close[i] > R1_aligned[i] or rsi[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0