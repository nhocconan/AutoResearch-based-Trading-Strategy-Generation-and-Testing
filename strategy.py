# 1d_KAMA_RSI_Chop_Filter_v2
# Hypothesis: Daily KAMA adapts to market noise, capturing trends while avoiding whipsaws. RSI filters overbought/oversold conditions, and Choppiness Index identifies trending vs ranging markets. Works in bull markets (KAMA up, RSI<70, trending) and bear markets (KAMA down, RSI>30, trending). Target: 10-25 trades/year per symbol.

name = "1d_KAMA_RSI_Chop_Filter_v2"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter (requires 2 extra weeks for confirmation)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=2)
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over 10 periods
    # Fix dimensions: volatility needs same length as change
    volatility = pd.Series(close).rolling(window=10).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([np.array([np.nan]), rsi])
    
    # Calculate Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0, 
                    100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(14), 
                    50)
    # Fix: rolling sum of ATR
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop = np.where((max_high - min_low) == 0, 50, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 10)  # KAMA(30), RSI(14), CHOP(14)
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising, RSI<70 (not overbought), trending market (CHOP<61.8), above weekly EMA34
            if (kama[i] > kama[i-1] and 
                rsi[i] < 70 and 
                chop[i] < 61.8 and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI>30 (not oversold), trending market (CHOP<61.8), below weekly EMA34
            elif (kama[i] < kama[i-1] and 
                  rsi[i] > 30 and 
                  chop[i] < 61.8 and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA falling OR RSI>70 (overbought) OR choppy market (CHOP>61.8)
            if (kama[i] < kama[i-1] or 
                rsi[i] > 70 or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA rising OR RSI<30 (oversold) OR choppy market (CHOP>61.8)
            if (kama[i] > kama[i-1] or 
                rsi[i] < 30 or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals