#!/usr/bin/env python3
"""
1d_KAMA_Regime_With_Volume_Confirmation
Hypothesis: 1d KAMA trend direction filtered by weekly choppiness regime and daily volume spike.
Long when KAMA rising, CHOP < 38.2 (trending), and volume > 1.5x 20-day MA.
Short when KAMA falling, CHOP < 38.2, and volume spike.
Uses ATR trailing stop (2.5) and discrete sizing (0.30) to limit trades (~10-25/year).
Designed for BTC/ETH to work in bull/bear via adaptive trend + regime filter + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for KAMA and volume MA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility correctly: sum of absolute daily changes over ER period
    er_period = 10
    volatility_sum = np.zeros_like(close_1d)
    for i in range(er_period, len(close_1d)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close_1d[i-er_period+1:i+1])))
    # Avoid division by zero
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    # Smoothing constants
    fastest = 2.0 / (2 + 1)
    slowest = 2.0 / (30 + 1)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # 1d volume MA (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1w data for choppiness regime
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Choppiness Index (14-period)
    chop_period = 14
    atr_1w = np.zeros_like(close_1w)
    tr_1w = np.zeros_like(close_1w)
    for i in range(1, len(close_1w)):
        tr_1w[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    # First TR
    tr_1w[0] = high_1w[0] - low_1w[0]
    # ATR
    for i in range(chop_period, len(close_1w)):
        atr_1w[i] = np.mean(tr_1w[i-chop_period+1:i+1])
    # Sum of ATR over period
    sum_atr = np.zeros_like(close_1w)
    for i in range(chop_period, len(close_1w)):
        sum_atr[i] = np.sum(atr_1w[i-chop_period+1:i+1])
    # Choppiness: 100 * log10(sum(ATR) / (max(high)-min(low)) * sqrt(period)) / log10(sqrt(period))
    max_high = np.zeros_like(close_1w)
    min_low = np.zeros_like(close_1w)
    for i in range(chop_period, len(close_1w)):
        max_high[i] = np.max(high_1w[i-chop_period+1:i+1])
        min_low[i] = np.min(low_1w[i-chop_period+1:i+1])
    range_hl = max_high - min_low
    # Avoid division by zero
    chop = np.where(
        (range_hl > 0) & (sum_atr > 0),
        100 * np.log10(sum_atr / range_hl * np.sqrt(chop_period)) / np.log10(np.sqrt(chop_period)),
        50  # neutral if undefined
    )
    
    # Align all HTF data to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # ATR for stop loss (14-period on 1d)
    tr_1d = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need KAMA, volume MA, chop, ATR
    start_idx = max(50, 20, 14, 30)  # KAMA needs lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: KAMA rising, chop < 38.2 (trending), volume spike
            kama_rising = kama_aligned[i] > kama_aligned[i-1]
            vol_spike = volume[i] > (1.5 * vol_ma_20_aligned[i])
            chop_low = chop_aligned[i] < 38.2
            
            if kama_rising and vol_spike and chop_low:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short: KAMA falling, chop < 38.2 (trending), volume spike
            elif (not kama_rising) and vol_spike and chop_low:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: KAMA falling OR chop > 61.8 (choppy) OR ATR stoploss hit
            kama_falling = kama_aligned[i] < kama_aligned[i-1]
            chop_high = chop_aligned[i] > 61.8
            if kama_falling or chop_high or (curr_close < entry_price - 2.5 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: KAMA rising OR chop > 61.8 (choppy) OR ATR stoploss hit
            if kama_rising or chop_high or (curr_close > entry_price + 2.5 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Regime_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0