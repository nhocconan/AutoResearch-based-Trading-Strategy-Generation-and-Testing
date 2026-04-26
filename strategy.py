#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter_and_Chop_Regime_v2
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
filtered by RSI extremes and Choppiness Index regime to avoid whipsaws. Enter long when KAMA turns up,
RSI < 30 (oversold), and market is trending (CHOP < 38.2). Enter short when KAMA turns down,
RSI > 70 (overbought), and market is trending. Exit on opposite signal or regime shift to ranging.
Designed to capture sustained trends in both bull and bear markets while avoiding choppy periods.
Target: 15-25 trades/year per symbol to minimize fee drag.
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
    
    # Get 1d data for KAMA, RSI, and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on 1d
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([[np.nan]*10, change])
    volatility = np.concatenate([[np.nan]*10, volatility])
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=np.float64)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (it's already on 1d)
    kama_aligned = kama  # Already on 1d
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average
    avg_gain = np.full_like(close_1d, np.nan, dtype=np.float64)
    avg_loss = np.full_like(close_1d, np.nan, dtype=np.float64)
    avg_gain[13] = np.mean(gain[1:14])  # 14-period average starting at index 1
    avg_loss[13] = np.mean(loss[1:14])
    # Wilder smoothing
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = rsi  # Already on 1d
    
    # Calculate Choppiness Index on 1d
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # ATR(14)
    atr_14 = np.full_like(close_1d, np.nan, dtype=np.float64)
    for i in range(14, len(close_1d)):
        atr_14[i] = np.mean(tr[i-13:i+1])  # Simple mean for ATR
    
    # Sum of ATR over 14 periods
    sum_atr_14 = np.full_like(close_1d, np.nan, dtype=np.float64)
    for i in range(27, len(close_1d)):  # 14+14-1
        sum_atr_14[i] = np.sum(atr_14[i-13:i+1])
    
    # Max(high) - Min(low) over 14 periods
    max_high_14 = np.full_like(close_1d, np.nan, dtype=np.float64)
    min_low_14 = np.full_like(close_1d, np.nan, dtype=np.float64)
    for i in range(13, len(close_1d)):
        max_high_14[i] = np.max(high[i-13:i+1])
        min_low_14[i] = np.min(low[i-13:i+1])
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index
    chop = np.full_like(close_1d, 50.0, dtype=np.float64)  # Default neutral
    for i in range(27, len(close_1d)):  # Need 28 values (0-indexed to 27)
        if sum_atr_14[i] > 0 and range_14[i] > 0:
            chop[i] = 100 * np.log10(sum_atr_14[i] / range_14[i]) / np.log10(14)
    
    chop_aligned = chop  # Already on 1d
    
    # Get 1w EMA20 for higher timeframe trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full_like(close_1w, np.nan, dtype=np.float64)
    for i in range(19, len(close_1w)):
        ema_20_1w[i] = np.mean(close_1w[i-19:i+1])  # Simple mean for EMA approximation
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), RSI (14), ATR (14+14), chop (14+14), 1w EMA (20)
    start_idx = max(10, 14, 27, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 1w EMA20
        price_above_1w_ema = close[i] > ema_20_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_20_1w_aligned[i]
        
        # KAMA direction: slope over 2 periods
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI conditions
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Regime filter: trending market (CHOP < 38.2)
        trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: KAMA rising + RSI oversold + trending + price above 1w EMA
            long_signal = kama_rising and rsi_oversold and trending and price_above_1w_ema
            
            # Short: KAMA falling + RSI overbought + trending + price below 1w EMA
            short_signal = kama_falling and rsi_overbought and trending and price_below_1w_ema
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA falling OR RSI overbought OR shift to ranging (CHOP > 50)
            if (kama_falling or rsi_overbought or chop_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA rising OR RSI oversold OR shift to ranging (CHOP > 50)
            if (kama_rising or rsi_oversold or chop_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter_and_Chop_Regime_v2"
timeframe = "1d"
leverage = 1.0