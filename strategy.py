# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop
Hypothesis: On daily timeframe, enter long when KAMA indicates uptrend, RSI is not overbought, and market is not choppy. Enter short when KAMA indicates downtrend, RSI is not oversold, and market is not choppy. Uses Choppiness Index as regime filter to avoid whipsaws in sideways markets. Designed for low trade frequency (~10-20/year) to minimize fee decay and work in both bull and bear markets.
"""

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
    
    # Get 1w data for trend filter and regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on weekly data
    def calculate_kama(close_prices, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close_prices, n=length))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        er = np.zeros_like(close_prices)
        er[length:] = change / np.where(volatility[length:] == 0, 1, volatility[length:])
        
        # Smoothing Constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA
        kama = np.zeros_like(close_prices)
        kama[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    # Calculate Choppiness Index on weekly data
    def calculate_chop(high_prices, low_prices, close_prices, length=14):
        atr = np.zeros_like(close_prices)
        tr1 = high_prices[1:] - low_prices[1:]
        tr2 = np.abs(high_prices[1:] - close_prices[:-1])
        tr3 = np.abs(low_prices[1:] - close_prices[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR calculation
        atr[length:] = np.nansum(tr.reshape(-1, length), axis=1) / length
        
        # Chop calculation
        max_high = np.zeros_like(close_prices)
        min_low = np.zeros_like(close_prices)
        for i in range(length, len(close_prices)):
            max_high[i] = np.max(high_prices[i-length+1:i+1])
            min_low[i] = np.min(low_prices[i-length+1:i+1])
        
        chop = np.full_like(close_prices, 50.0)
        for i in range(length, len(close_prices)):
            if np.sum(atr[i-length+1:i+1]) > 0:
                chop[i] = 100 * np.log10(np.sum(atr[i-length+1:i+1]) / 
                                          (max_high[i] - min_low[i])) / np.log10(length)
        return chop
    
    # Calculate indicators on weekly data
    kama = calculate_kama(df_1w['close'].values, length=10, fast=2, slow=30)
    chop = calculate_chop(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, length=14)
    
    # Calculate RSI on daily data for entry timing
    def calculate_rsi(close_prices, length=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        avg_gain[length:] = np.nanmean(gain.reshape(-1, length), axis=1)
        avg_loss[length:] = np.nanmean(loss.reshape(-1, length), axis=1)
        
        rs = np.where(avg_loss[length:] == 0, 100, avg_gain[length:] / avg_loss[length:])
        rsi = np.full_like(close_prices, 50.0)
        rsi[length:] = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, length=14)
    
    # Align all weekly data to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition: price relative to KAMA
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # RSI conditions: not overbought/oversold
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Regime filter: not choppy (Choppiness Index < 61.8 = trending)
        not_choppy = chop_aligned[i] < 61.8
        
        # Entry conditions
        long_entry = price_above_kama and rsi_not_overbought and not_choppy
        short_entry = price_below_kama and rsi_not_oversold and not_choppy
        
        # Exit conditions: opposite signal or choppy market
        long_exit = price_below_kama or chop_aligned[i] >= 61.8
        short_exit = price_above_kama or chop_aligned[i] >= 61.8
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop"
timeframe = "1d"
leverage = 1.0