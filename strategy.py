#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA + RSI + Chop regime filter. 
# KAMA adapts to market noise, reducing false signals in chop. 
# RSI avoids overextended entries. Chop filter ensures we only trend-follow when market is trending (CHOP < 38.2) or mean-revert when choppy (CHOP > 61.8).
# Works in bull (KAMA up, RSI pullbacks) and bear (KAMA down, bounces). Target: 20-40 trades/year.

def calculate_kama(close, er_period=10, fast_ema=2, slow_ema=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    er = np.zeros_like(close)
    for i in range(len(close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    sc = np.zeros_like(close)
    fast_sc = 2 / (fast_ema + 1)
    slow_sc = 2 / (slow_ema + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.zeros_like(close)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_chop(high, low, close, period=14):
    """Choppiness Index"""
    atr = np.zeros_like(close)
    tr = np.zeros_like(close)
    
    for i in range(len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1] if i>0 else close[i]), abs(low[i] - close[i-1] if i>0 else close[i]))
    
    for i in range(len(close)):
        if i >= period:
            atr[i] = np.sum(tr[i-period+1:i+1]) / period
        else:
            atr[i] = np.nan
    
    chop = np.zeros_like(close)
    for i in range(len(close)):
        if i >= period and not np.isnan(atr[i]) and atr[i] > 0:
            highest_high = np.max(high[i-period+1:i+1])
            lowest_low = np.min(low[i-period+1:i+1])
            if highest_high > lowest_low:
                chop[i] = 100 * np.log10(atr[i] * period / (highest_high - lowest_low)) / np.log10(period)
            else:
                chop[i] = 50
        else:
            chop[i] = np.nan
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1w data for Chop regime filter ===
    df_1w = get_htf_data(prices, '1w')
    chop_1w = calculate_chop(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # === 1d data for KAMA and RSI ===
    df_1d = get_htf_data(prices, '1d')
    kama_1d = calculate_kama(df_1d['close'].values, 10, 2, 30)
    rsi_1d = calculate_rsi(df_1d['close'].values, 14)
    
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0
    warmup = 50
    
    for i in range(warmup, n):
        if (np.isnan(chop_1w_aligned[i]) or 
            np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop = chop_1w_aligned[i]
        kama = kama_1d_aligned[i]
        rsi = rsi_1d_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price > KAMA (uptrend) + RSI not overbought + chop regime allows trend
            if price > kama and rsi < 70 and (chop < 38.2 or chop > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend) + RSI not oversold + chop regime allows trend
            elif price < kama and rsi > 30 and (chop < 38.2 or chop > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA or RSI overextended
            if price < kama or rsi > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA or RSI overextended
            if price > kama or rsi < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_ChopRegime"
timeframe = "12h"
leverage = 1.0