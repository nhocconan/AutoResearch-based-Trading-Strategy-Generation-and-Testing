#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d RSI mean reversion + volume spike
# In choppy markets (high CHOP), RSI extremes revert to mean. Uses daily RSI(14) on 4h chart.
# Volume spike filters low-probability signals. Designed for low-frequency trades to work in both bull/bear markets.

name = "4h_Chop_RSI14_1d_Volume"
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
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:13] = np.nan
    
    # Align RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Choppiness Index on 4h data (14-period)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(14)
    atr_sum = np.zeros(n)
    atr_sum[13] = np.sum(tr[1:14])
    for i in range(14, n):
        atr_sum[i] = atr_sum[i-1] - atr_sum[i-1]/14 + tr[i]
    atr = atr_sum / 14
    
    # True Range sum for denominator
    tr_sum = np.zeros(n)
    tr_sum[13] = np.sum(tr[1:14])
    for i in range(14, n):
        tr_sum[i] = tr_sum[i-1] - tr_sum[i-1]/14 + tr[i]
    
    # Highest high and lowest low over 14 periods
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(n):
        if i < 13:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high[i-13:i+1])
            lowest_low[i] = np.min(low[i-13:i+1])
    
    # Chop = 100 * log10(sum(tr) / (HH - LL)) / log10(14)
    chop = np.zeros(n)
    for i in range(13, n):
        if highest_high[i] > lowest_low[i] and not np.isnan(tr_sum[i]):
            chop[i] = 100 * np.log10(tr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = np.nan
    
    # Volume spike (2x 20-period EMA)
    vol_ma = np.zeros(n)
    vol_ma[19] = np.mean(volume[0:20])
    for i in range(20, n):
        vol_ma[i] = vol_ma[i-1] * 0.9 + volume[i] * 0.1
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure RSI and other indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop > 61.8 indicates ranging/choppy market (good for mean reversion)
        is_choppy = chop[i] > 61.8
        
        if position == 0:
            # Enter long: RSI oversold (<30) in choppy market with volume spike
            if (rsi_1d_aligned[i] < 30 and is_choppy and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought (>70) in choppy market with volume spike
            elif (rsi_1d_aligned[i] > 70 and is_choppy and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (>50) or chop ends
            if (rsi_1d_aligned[i] > 50 or chop[i] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral (<50) or chop ends
            if (rsi_1d_aligned[i] < 50 or chop[i] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals