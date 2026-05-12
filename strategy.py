#!/usr/bin/env python3
name = "4h_KAMA_Direction_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation
    def calculate_kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI calculation
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Choppiness Index calculation
    def calculate_chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=length, min_periods=length).sum().values
        highest_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        chop = 100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(length)
        return chop
    
    # Load daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily KAMA for trend filter
    kama_1d = calculate_kama(close_1d, length=10, fast=2, slow=30)
    kama_1d_dir = np.where(kama_1d > np.roll(kama_1d, 1), 1, -1)
    kama_1d_dir[0] = 1  # initialize
    kama_1d_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_dir)
    
    # Calculate daily RSI for momentum filter
    rsi_1d = calculate_rsi(close_1d, length=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 4h Choppiness Index for regime filter
    chop = calculate_chop(high, low, close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_dir_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + chop < 61.8 (trending market)
            if kama_1d_dir_aligned[i] == 1 and rsi_1d_aligned[i] > 50 and chop[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + chop < 61.8 (trending market)
            elif kama_1d_dir_aligned[i] == -1 and rsi_1d_aligned[i] < 50 and chop[i] < 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down OR RSI < 40
            if kama_1d_dir_aligned[i] == -1 or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up OR RSI > 60
            if kama_1d_dir_aligned[i] == 1 or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals