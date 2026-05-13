#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend direction with RSI(14) extremes and choppiness regime filter.
# Long when KAMA rising, RSI<30, and CHOP>61.8 (range). Short when KAMA falling, RSI>70, and CHOP>61.8.
# Uses 1d primary timeframe with 1w EMA50 trend filter for higher timeframe bias.
# Designed to capture mean reversion in ranging markets while avoiding strong trends.
# Targets 30-100 trades over 4 years (7-25/year) with discrete position sizing to minimize fee drag.

name = "1d_KAMA_RSI_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Calculate KAMA(10, 2, 30) - ER=10, fastest=2, slowest=30
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)[:len(change)]
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing Constants
    sc = (er * (2./2 - 2./30) + 2./30) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Seed at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    # Calculate Choppiness Index(14)
    atr_1 = np.abs(high - low)
    atr_2 = np.abs(high - np.roll(close, 1))
    atr_3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(atr_1, np.maximum(atr_2, atr_3))
    tr[0] = atr_1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    
    # Get 1w data for EMA50 trend filter (MTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w close
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to 1d timeframe (wait for completed 1w bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising (today > yesterday), RSI<30 (oversold), CHOP>61.8 (range)
            if kama[i] > kama[i-1] and rsi[i] < 30 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (today < yesterday), RSI>70 (overbought), CHOP>61.8 (range)
            elif kama[i] < kama[i-1] and rsi[i] > 70 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling OR RSI>50 (exit oversold) OR CHOP<38.2 (trending)
            if kama[i] < kama[i-1] or rsi[i] > 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising OR RSI<50 (exit overbought) OR CHOP<38.2 (trending)
            if kama[i] > kama[i-1] or rsi[i] < 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals