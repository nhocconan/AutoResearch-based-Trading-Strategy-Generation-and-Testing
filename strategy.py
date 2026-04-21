#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d KAMA trend direction + RSI filter + chop regime filter.
KAMA adapts to market conditions - fast in trends, slow in ranging markets.
RSI filters for overbought/oversold conditions within the trend.
Chop filter avoids whipsaws in ranging markets.
Target: 20-30 trades/year to minimize fee drag while capturing major trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for KAMA, RSI, and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d KAMA (Adaptive Moving Average)
    close_1d = df_1d['close'].values
    direction = np.abs(np.diff(close_1d, n=10, prepend=close_1d[:10]))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1, prepend=close_1d[0])), axis=0)
    # Handle 1D case for volatility calculation
    if volatility.ndim == 0:
        volatility = np.full_like(close_1d, volatility)
    else:
        volatility = np.concatenate([np.full(10, volatility[0]) if len(volatility) > 10 else np.full(len(close_1d), volatility[0]), volatility])
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    efficiency_ratio = np.where(volatility > 0, direction / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    if len(gain) > 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr_1d = np.zeros(len(close_1d))
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[tr[0]] if len(tr) > 0 else [0], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).sum().values / 14
    
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        if atr_1d[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(sum(atr_1d[i-13:i+1]) / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    # Align all indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop_aligned[i]
        
        # Determine trend direction from KAMA
        # Uptrend: price > KAMA, Downtrend: price < KAMA
        if position == 0:
            # Enter long: price above KAMA (uptrend) + RSI not overbought + chop not extreme
            if (price_close > kama_aligned[i] and 
                rsi_val < 70 and 
                chop_val < 61.8):  # Not in strong chop (trending market)
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend) + RSI not oversold + chop not extreme
            elif (price_close < kama_aligned[i] and 
                  rsi_val > 30 and 
                  chop_val < 61.8):  # Not in strong chop (trending market)
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit: price crosses below KAMA OR RSI overbought OR chop extreme
                if (price_close < kama_aligned[i] or 
                    rsi_val > 75 or 
                    chop_val > 61.8):
                    exit_signal = True
            else:  # Short position
                # Exit: price crosses above KAMA OR RSI oversold OR chop extreme
                if (price_close > kama_aligned[i] or 
                    rsi_val < 25 or 
                    chop_val > 61.8):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_KAMA_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0