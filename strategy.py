#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Hypothesis: Daily trend following using Kaufman Adaptive Moving Average (KAMA) direction,
# filtered by RSI for momentum and Choppiness Index to avoid range-bound markets.
# Long when KAMA trending up, RSI > 50, and Choppiness < 61.8 (trending regime).
# Short when KAMA trending down, RSI < 50, and Choppiness < 61.8.
# Exit when any condition fails.
# Uses 1-week trend filter to ensure alignment with higher timeframe momentum.
# Target: 10-25 trades/year to minimize fee decay while capturing sustained trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === KAMA (10-period ER, 2/30 SC) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # needs rolling sum
    
    # Correct ER calculation
    er = np.zeros(n)
    for i in range(n):
        if i < 10:
            er[i] = np.nan
        else:
            price_change = np.abs(close[i] - close[i-10])
            vol_sum = 0.0
            for j in range(i-9, i+1):
                vol_sum += np.abs(close[j] - close[j-1])
            er[i] = price_change / vol_sum if vol_sum != 0 else 0
    
    # Smoothing Constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    if not np.isnan(sc[0]):
        kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14-period) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    if n > 14:
        avg_gain[14] = np.mean(gain[1:15])  # gains[1] to gains[14]
        avg_loss[14] = np.mean(loss[1:15])
        
        for i in range(15, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14-period) ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    
    # ATR (14-period)
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])  # simple mean of last 14 TR
    
    # Sum of ATR over 14 periods
    atr_sum = np.full(n, np.nan)
    for i in range(27, n):  # need 14 ATR values
        atr_sum[i] = np.nansum(atr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(13, n):
        highest_high[i] = np.nanmax(high[i-13:i+1])
        lowest_low[i] = np.nanmin(low[i-13:i+1])
    
    # Chop = 100 * log10(ATR_sum / (HH - LL)) / log10(14)
    chop = np.full(n, np.nan)
    for i in range(27, n):
        if atr_sum[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = np.nan
    
    # === Higher Timeframe Filter: 1-week trend ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Simple 20-period SMA for 1w trend
    sma_1w = np.full(len(close_1w), np.nan)
    for i in range(19, len(close_1w)):
        sma_1w[i] = np.mean(close_1w[i-19:i+1])
    
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # === Signal Generation ===
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # start after warmup
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(sma_1w_aligned[i])):
            if position != 0:
                pass  # hold position
            else:
                signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        kama_prev = kama[i-1] if i > 0 else kama_val
        rsi_val = rsi[i]
        chop_val = chop[i]
        sma_1w_val = sma_1w_aligned[i]
        price = close[i]
        
        # KAMA direction: rising if current > previous
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        if position == 1:  # Long
            # Exit: KAMA turns down OR RSI < 50 OR Chop > 61.8 (ranging) OR price < 1w SMA
            if (not kama_rising) or rsi_val < 50 or chop_val > 61.8 or price < sma_1w_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: KAMA turns up OR RSI > 50 OR Chop > 61.8 OR price > 1w SMA
            if (not kama_falling) or rsi_val > 50 or chop_val > 61.8 or price > sma_1w_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions
            # Long: KAMA rising, RSI > 50, Chop < 61.8, price > 1w SMA
            # Short: KAMA falling, RSI < 50, Chop < 61.8, price < 1w SMA
            if kama_rising and rsi_val > 50 and chop_val < 61.8 and price > sma_1w_val:
                position = 1
                signals[i] = 0.25
            elif kama_falling and rsi_val < 50 and chop_val < 61.8 and price < sma_1w_val:
                position = -1
                signals[i] = -0.25
    
    return signals