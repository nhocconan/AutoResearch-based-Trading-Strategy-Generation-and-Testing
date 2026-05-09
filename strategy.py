#!/usr/bin/env python3
"""
Hypothesis: 4-hour timeframe with 1-day RSI(14) regime filter and 4-hour KAMA crossover.
In bullish regime (1d RSI > 50), go long when 4h KAMA crosses above 4h SMA(20).
In bearish regime (1d RSI < 50), go short when 4h KAMA crosses below 4h SMA(20).
Uses volume confirmation: require volume > 20-period average for entries.
Exits on opposite KAMA/SMA cross or volatility regime shift (1d RSI crosses 50).
Target: 20-50 trades per year with size 0.25.
"""

name = "4h_KAMA_SMA_Crossover_1dRSI_Regime_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day RSI(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_above_50 = rsi_1d_values > 50
    rsi_1d_below_50 = rsi_1d_values < 50
    rsi_1d_above_50_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_above_50)
    rsi_1d_below_50_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_below_50)
    
    # 4-hour KAMA(10, 2, 30) - using close prices
    if len(close) < 30:
        return np.zeros(n)
    
    # Efficiency ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Pad volatility to match length
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[29] = close[:30].mean()  # Seed with simple average
    for i in range(30, len(close)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 4-hour SMA(20)
    if len(close) < 20:
        return np.zeros(n)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma
    
    # Alignment for 4h indicators
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    sma_20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), sma_20)
    volume_confirm_aligned = align_htf_to_ltf(prices, pd.DataFrame({'volume': volume}), volume_confirm)
    
    # Crossover signals
    kama_above_sma = kama_aligned > sma_20_aligned
    kama_below_sma = kama_aligned < sma_20_aligned
    
    # Previous values for crossover detection
    kama_above_sma_prev = np.concatenate([[False], kama_above_sma[:-1]])
    kama_below_sma_prev = np.concatenate([[False], kama_below_sma[:-1]])
    
    # Entry conditions
    long_entry = (kama_above_sma & ~kama_above_sma_prev & 
                  rsi_1d_above_50_aligned & volume_confirm_aligned)
    short_entry = (kama_below_sma & ~kama_below_sma_prev & 
                   rsi_1d_below_50_aligned & volume_confirm_aligned)
    
    # Exit conditions: opposite cross or regime change
    long_exit = (kama_below_sma & ~kama_below_sma_prev) | (~rsi_1d_above_50_aligned)
    short_exit = (kama_above_sma & ~kama_above_sma_prev) | (~rsi_1d_below_50_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for KAMA and SMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(sma_20_aligned[i]) or
            np.isnan(rsi_1d_above_50_aligned[i]) or np.isnan(rsi_1d_below_50_aligned[i]) or
            np.isnan(volume_confirm_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            if long_entry[i]:
                signals[i] = 0.25
                position = 1
            elif short_entry[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            if long_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if short_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals