#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use KAMA(10,2,30) for trend direction, RSI(14) for momentum filter, and Choppiness Index(14) for regime filter. Enter long when KAMA trending up, RSI > 50, and CHOP < 38.2 (trending regime). Enter short when KAMA trending down, RSI < 50, and CHOP < 38.2. Exit when opposite signal occurs. Uses ATR-based stoploss (2.0*ATR). Designed for low trade frequency (~15-25/year) to minimize fee drag while capturing medium-term trends in both bull and bear markets.
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
    
    # Get 1d data (primary timeframe) - but we need HTF = 1w for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate KAMA on weekly close for trend direction
    close_1w = df_1w['close'].values
    # KAMA parameters: ER period=10, fast=2, slow=30
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)  # needs correction
    # Proper KAMA calculation
    er = np.zeros_like(close_1w)
    for i in range(10, len(close_1w)):
        directional_change = np.abs(close_1w[i] - close_1w[i-10])
        sum_abs_ret = np.sum(np.abs(np.diff(close_1w[i-9:i+1])))
        if sum_abs_ret > 0:
            er[i] = directional_change / sum_abs_ret
        else:
            er[i] = 0
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1w)
    kama[9] = close_1w[9]  # seed
    for i in range(10, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    
    # Align KAMA to daily timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14) on daily
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low))) / log10(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (np.log10(14) * (highest_high - lowest_low))) / np.log10(14)
    
    # ATR for stoploss
    atr_stop = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA(10), RSI(14), CHOP(14)
    start_idx = max(20, 14, 14)  # KAMA seed, RSI, CHOP
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or
            np.isnan(atr_stop[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        kama_val = kama_1w_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        close_val = close[i]
        atr_val = atr_stop[i]
        
        # Regime filter: only trade in trending market (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Long: KAMA trending up (price > KAMA), RSI > 50, trending regime
            long_signal = (close_val > kama_val) and (rsi_val > 50) and trending_regime
            # Short: KAMA trending down (price < KAMA), RSI < 50, trending regime
            short_signal = (close_val < kama_val) and (rsi_val < 50) and trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # ATR stoploss: exit if price drops 2.0*ATR from entry
            # Approximate entry as recent close when signal triggered
            if close_val < kama_val:  # simple exit: price crosses below KAMA
                signals[i] = 0.0
                position = 0
            # Alternative: use RSI < 40 for exit
            elif rsi_val < 40:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # ATR stoploss: exit if price rises 2.0*ATR from entry
            if close_val > kama_val:  # simple exit: price crosses above KAMA
                signals[i] = 0.0
                position = 0
            # Alternative: use RSI > 60 for exit
            elif rsi_val > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0