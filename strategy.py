#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI filter and ATR stoploss
# - Uses 1w HTF for regime filter: only trade when price > 1w EMA200 (bull) or < 1w EMA200 (bear)
# - KAMA(10,2,30) on 1d for adaptive trend direction
# - RSI(14) filter: avoid overbought/oversold extremes (long when RSI<70, short when RSI>30)
# - ATR(14) trailing stop: exit when price moves 2.5x ATR against position
# - Fixed position size 0.25 to control drawdown
# - Target: 30-100 trades over 4 years (7-25/year)
# - Works in bull/bear by aligning with higher timeframe trend

name = "1d_kama_rsi_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for regime filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute KAMA on 1d
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 0 else np.sum(np.abs(np.diff(close)))
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=np.float64)
        kama[er_length] = close[er_length]
        for i in range(er_length + 1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    
    # Pre-compute RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align with close
    
    # Pre-compute ATR(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema200_1w_aligned[i]) or
            np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.5x ATR from highest high
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.5x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: KAMA trend + RSI filter + 1w regime
            bull_regime = close[i] > ema200_1w_aligned[i]
            bear_regime = close[i] < ema200_1w_aligned[i]
            
            # Long: price above KAMA (uptrend), RSI not overbought, bull regime
            if close[i] > kama[i] and rsi[i] < 70 and bull_regime:
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            # Short: price below KAMA (downtrend), RSI not oversold, bear regime
            elif close[i] < kama[i] and rsi[i] > 30 and bear_regime:
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals