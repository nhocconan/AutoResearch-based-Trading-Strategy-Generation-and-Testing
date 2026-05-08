#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA with RSI and Chop filter
# Long when KAMA trending up, RSI < 50 (dip in uptrend), Chop > 61.8 (range)
# Short when KAMA trending down, RSI > 50 (bounce in downtrend), Chop > 61.8 (range)
# Uses daily timeframe for signal generation with Chop regime filter to avoid trending markets
# Targets 30-100 total trades over 4 years (7-25/year) for low fee drag and high win rate

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for signal generation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    # ER (Efficiency Ratio) = |change| / sum(|changes|)
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(change) > 1 else 1
    er = np.concatenate([[0], np.abs(np.diff(close_1d))]) / (np.concatenate([[1]], np.sum(np.abs(np.diff(close_1d)))) + 1e-10)
    # Simplified ER calculation for stability
    price_change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    total_change = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])))
    volatility_sum = np.convolve(price_change, np.ones(10), 'same')  # 10-period volatility
    volatility_sum = np.where(volatility_sum == 0, 1, volatility_sum)
    er = price_change / volatility_sum
    # Smoothing constants
    sc = (er * (0.6665 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (no additional delay needed)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI on daily close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)  # Fill NaN with neutral
    
    # Align RSI to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Chop index on daily data
    atr_period = 14
    tr1 = np.abs(np.subtract(df_1d['high'].values, df_1d['low'].values))
    tr2 = np.abs(np.subtract(df_1d['high'].values, np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])))
    tr3 = np.abs(np.subtract(df_1d['low'].values, np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(df_1d['high'].values).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(df_1d['low'].values).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_max_min = max_high - min_low
    range_max_min = np.where(range_max_min == 0, 1e-10, range_max_min)
    
    chop = 100 * np.log10(atr * atr_period / range_max_min) / np.log10(atr_period)
    chop = np.where(np.isnan(chop), 50, chop)  # Fill NaN with neutral
    
    # Align Chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Weekly trend filter: price above/below weekly EMA20
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema20_1w_val = ema20_1w_aligned[i]
        
        if position == 0:
            # Enter long: KAMA trending up, RSI < 50 (dip), Chop > 61.8 (range)
            if close_val > kama_val and rsi_val < 50 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA trending down, RSI > 50 (bounce), Chop > 61.8 (range)
            elif close_val < kama_val and rsi_val > 50 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down or Chop < 38.2 (trending)
            if close_val < kama_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up or Chop < 38.2 (trending)
            if close_val > kama_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals