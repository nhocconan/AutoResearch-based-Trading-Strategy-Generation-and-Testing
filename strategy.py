#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_RSI_Chop_Regime_v1
Hypothesis: Daily KAMA trend direction filters RSI mean-reversion entries, with Choppiness Index regime filter to avoid whipsaws.
Long when KAMA rising (bull trend) + RSI<30 + Chop>61.8 (range). Short when KAMA falling (bear trend) + RSI>70 + Chop>61.8.
ATR-based stoploss (2.0x) and discrete position sizing (0.25) to manage drawdown and fee drag.
Designed for low trade frequency (<25/year) and works in both bull and bear markets via trend filter + regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend context)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly EMA20 for higher timeframe trend filter ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Daily KAMA for primary trend ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of abs changes
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    er = np.concatenate([np.full(9, np.nan), er])  # align to original length
    
    # Smoothing constants
    fast_sc = np.log(2 / (2 + 1))
    slow_sc = np.log(2 / (30 + 1))
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = np.where(np.isnan(sc), slow_sc, sc)  # default to slow when ER NaN
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Daily RSI(14) for mean reversion ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])  # align
    
    # === Daily Choppiness Index(14) for regime filter ===
    atr_1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_sum = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high - lowest_low) != 0, 
                    100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14), 
                    50)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])  # align
    
    # === ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Trend filter: price relative to weekly EMA20
            uptrend = price > ema_20_1w_aligned[i]
            downtrend = price < ema_20_1w_aligned[i]
            
            # KAMA slope (trend strength)
            kama_rising = kama[i] > kama[i-1]
            kama_falling = kama[i] < kama[i-1]
            
            # Mean reversion signals
            rsi_oversold = rsi[i] < 30
            rsi_overbought = rsi[i] > 70
            
            # Regime filter: choppy market (range-bound)
            choppy = chop[i] > 61.8
            
            # Entry logic: trend-aligned mean reversion in choppy regime
            if uptrend and kama_rising and rsi_oversold and choppy:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif downtrend and kama_falling and rsi_overbought and choppy:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Stoploss: 2.0x ATR
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: trend reversal or regime change
            elif kama[i] < kama[i-1] or chop[i] < 38.2:  # trend weakening or trending regime
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Stoploss: 2.0x ATR
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: trend reversal or regime change
            elif kama[i] > kama[i-1] or chop[i] < 38.2:  # trend weakening or trending regime
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_Filter_RSI_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0