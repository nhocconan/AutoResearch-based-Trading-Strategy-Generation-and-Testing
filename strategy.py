#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop Filter
# KAMA adapts to market noise, reducing whipsaw in choppy conditions.
# RSI provides overbought/oversold signals for mean reversion.
# Chop filter (Choppiness Index) identifies ranging markets where mean reversion works best.
# In trending markets (low chop), we follow KAMA direction.
# In ranging markets (high chop), we mean revert at RSI extremes.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Chop filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # temporary fix, will compute properly below
    # Recalculate volatility properly
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Actually, let's compute ER and volatility correctly
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # ER for period 10
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if volatility[i] != 0:
            er[i] = np.abs(close[i] - close[i-10]) / (volatility[i] - volatility[i-10])
        else:
            er[i] = 0
    # SC = [ER * (fastest - slowest) + slowest]^2
    fastest = 2/(2+1)   # for EMA 2
    slowest = 2/(30+1)  # for EMA 30
    sc = (er * (fastest - slowest) + slowest) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index on weekly data
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    atr_list = []
    for i in range(len(df_1w)):
        tr = max(
            df_1w['high'].iloc[i] - df_1w['low'].iloc[i],
            abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1]) if i > 0 else 0,
            abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1]) if i > 0 else 0
        )
        atr_list.append(tr)
    atr = np.array(atr_list)
    
    chop_period = 14
    chop = np.full(len(df_1w), np.nan)
    for i in range(chop_period, len(df_1w)):
        atr_sum = np.sum(atr[i-chop_period+1:i+1])
        high_max = np.max(df_1w['high'].iloc[i-chop_period+1:i+1].values)
        low_min = np.min(df_1w['low'].iloc[i-chop_period+1:i+1].values)
        if high_max != low_min:
            chop[i] = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(chop_period)
        else:
            chop[i] = 50  # avoid division by zero
    
    # Align Chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Align KAMA and RSI (already daily)
    # No alignment needed for KAMA and RSI as they're calculated on close
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # In ranging market (high chop), mean revert at RSI extremes
            if chop_val > 61.8:  # ranging market
                if rsi_val < 30:  # oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi_val > 70:  # overbought
                    signals[i] = -0.25
                    position = -1
            # In trending market (low chop), follow KAMA direction
            else:  # trending market
                if close_val > kama_val:  # uptrend
                    signals[i] = 0.25
                    position = 1
                elif close_val < kama_val:  # downtrend
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or price crosses below KAMA
            if rsi_val > 70 or close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or price crosses above KAMA
            if rsi_val < 30 or close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals