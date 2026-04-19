#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_Filter_V2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on close
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Fix volatility calculation using rolling sum
    volatility_series = pd.Series(np.abs(np.diff(close, prepend=close[0])))
    volatility = volatility_series.rolling(window=er_period, min_periods=1).sum().values
    
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (fast_ema - 1) + 1) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Chop on weekly high/low
    chop_period = 14
    atr = np.maximum(np.maximum(df_1w['high'] - df_1w['low'], 
                                np.abs(df_1w['high'] - df_1w['close'].shift(1))),
                        np.abs(df_1w['low'] - df_1w['close'].shift(1)))
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(df_1w['high']).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(df_1w['low']).rolling(window=chop_period, min_periods=chop_period).min().values
    
    chop = np.where((highest_high - lowest_low) != 0, 
                    100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(chop_period), 
                    50)
    
    # Align indicators to daily
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Chop filter: only trade when chop < 61.8 (trending market)
        chop_filter = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long when price > KAMA and RSI > 50 in trending market
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short when price < KAMA and RSI < 50 in trending market
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and chop_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price < KAMA or RSI < 40
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price > KAMA or RSI > 60
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals