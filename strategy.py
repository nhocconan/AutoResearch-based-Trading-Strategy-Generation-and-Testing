#!/usr/bin/env python3
name = "1d_1w_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (HMA21)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly HMA(21) for trend filter
    close_1w = df_1w['close'].values
    n_hma = 21
    wama = np.zeros_like(close_1w)
    half_n = int(n_hma / 2)
    sqrt_n = int(np.sqrt(n_hma))
    
    for i in range(len(close_1w)):
        if i < n_hma - 1:
            wama[i] = np.nan
        else:
            # WMA of half period
            if i >= half_n - 1:
                wama_half = np.nansum(close_1w[i - half_n + 1:i + 1] * np.arange(1, half_n + 1)) / (half_n * (half_n + 1) / 2)
            else:
                wama_half = np.nan
            
            # WMA of full period
            wama_full = np.nansum(close_1w[i - n_hma + 1:i + 1] * np.arange(1, n_hma + 1)) / (n_hma * (n_hma + 1) / 2)
            
            # WAMA = 2*WMA(half) - WMA(full)
            wama[i] = 2 * wama_half - wama_full
    
    # HMA = WMA(WAMA, sqrt(n))
    hma_21 = np.full_like(close_1w, np.nan)
    for i in range(len(wama)):
        if i >= sqrt_n - 1 and not np.isnan(wama[i - sqrt_n + 1:i + 1]).any():
            hma_21[i] = np.nansum(wama[i - sqrt_n + 1:i + 1] * np.arange(1, sqrt_n + 1)) / (sqrt_n * (sqrt_n + 1) / 2)
    
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Calculate KAMA for direction (ER=10)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))).reshape(-1, 1)  # placeholder, will compute properly
    
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(len(close)):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            sum_abs_changes = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if sum_abs_changes > 0:
                er[i] = price_change / sum_abs_changes
            else:
                er[i] = 0
        else:
            er[i] = 0
    
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(len(close)):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (13) + gain[i]) / 14 if i > 0 else gain[i]
                avg_loss[i] = (avg_loss[i-1] * (13) + loss[i]) / 14 if i > 0 else loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Chopiness Index(14) for regime filter
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr_sum = np.zeros_like(close)
    for i in range(len(close)):
        if i < 14:
            atr_sum[i] = np.sum(tr[:i+1])
        else:
            atr_sum[i] = np.sum(tr[i-13:i+1])
    
    atr = atr_sum / 14
    
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    
    chop = np.zeros_like(close)
    for i in range(len(close)):
        if atr[i] > 0 and i >= 13:
            sum_tr = atr_sum[i]
            max_range = highest_high[i] - lowest_low[i]
            if max_range > 0:
                chop[i] = 100 * np.log10(sum_tr / max_range) / np.log10(14)
            else:
                chop[i] = 50
        else:
            chop[i] = 50
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(hma_21_aligned[i]) or 
            np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly HMA21 (uptrend), KAMA rising, RSI < 50, chop > 61.8 (range)
            if (close[i] > hma_21_aligned[i] and 
                kama[i] > kama[i-1] and 
                rsi[i] < 50 and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly HMA21 (downtrend), KAMA falling, RSI > 50, chop > 61.8 (range)
            elif (close[i] < hma_21_aligned[i] and 
                  kama[i] < kama[i-1] and 
                  rsi[i] > 50 and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly HMA21 or RSI > 70
            if close[i] < hma_21_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly HMA21 or RSI < 30
            if close[i] > hma_21_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals