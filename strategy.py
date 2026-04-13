#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter combined with 1d RSI mean reversion.
# - Choppiness Index (14) > 61.8 indicates ranging market (mean reversion regime)
# - RSI(14) on 1d timeframe for entry: long when RSI < 30, short when RSI > 70
# - Only trade in ranging regime to avoid whipsaws in trends
# - Works in both bull and bear markets by focusing on mean reversion in ranges
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4h data for Choppiness Index calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range calculation for Choppiness Index
    tr = np.zeros(len(close_4h))
    tr[0] = high_4h[0] - low_4h[0]
    for i in range(1, len(close_4h)):
        hl = high_4h[i] - low_4h[i]
        hc = abs(high_4h[i] - close_4h[i-1])
        lc = abs(low_4h[i] - close_4h[i-1])
        tr[i] = max(hl, hc, lc)
    
    # ATR(14) for Choppiness Index
    atr_14 = np.zeros(len(close_4h))
    atr_14[13] = np.mean(tr[0:14])
    for i in range(14, len(close_4h)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Choppiness Index: 100 * log(sum(ATR14)/log(14)) / log(high-low range over 14 periods)
    chop = np.full(len(close_4h), np.nan)
    for i in range(13, len(close_4h)):
        sum_atr = np.sum(atr_14[i-13:i+1])
        max_high = np.max(high_4h[i-13:i+1])
        min_low = np.min(low_4h[i-13:i+1])
        if max_high > min_low and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral value
    
    # Align Choppiness Index to 4h timeframe (already on 4h, but need alignment for safety)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    # 1-day data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # RSI(14) calculation
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[0:14])
    avg_loss[13] = np.mean(loss[0:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:14] = np.nan  # not enough data
    
    # Align RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(14, n):
        # Skip if any required data is not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        
        # Range regime: Choppiness Index > 61.8 indicates ranging market
        is_ranging = chop_val > 61.8
        
        if position == 0:
            # Long: RSI < 30 (oversold) in ranging market
            if is_ranging and rsi_val < 30:
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) in ranging market
            elif is_ranging and rsi_val > 70:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) or market starts trending
            if rsi_val > 50 or chop_val <= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) or market starts trending
            if rsi_val < 50 or chop_val <= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Choppiness_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0