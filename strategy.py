#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_KAMA_RSI_Chop_Filter_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA on 1d close
    close_1d = df_1d['close'].values
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    
    # Efficiency Ratio
    er = np.zeros_like(close_1d)
    for i in range(er_period, len(close_1d)):
        dir_change = np.abs(close_1d[i] - close_1d[i - er_period])
        total_ch = np.sum(np.abs(np.diff(close_1d[i - er_period:i + 1])))
        if total_ch > 0:
            er[i] = dir_change / total_ch
        else:
            er[i] = 0
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI on 1d close
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index on 1d
    chop_period = 14
    atr = np.zeros(len(high))
    for i in range(1, len(high)):
        atr[i] = max(
            high[i] - low[i],
            np.abs(high[i] - close[i-1]),
            np.abs(low[i] - close[i-1])
        )
    
    tr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    chop = np.zeros(len(close_1d))
    for i in range(chop_period-1, len(close_1d)):
        if tr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(chop_period)
        else:
            chop[i] = 50
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
            
        # KAMA direction: price > KAMA = uptrend, price < KAMA = downtrend
        kama_trend = 1 if close[i] > kama_aligned[i] else -1
        
        # RSI levels: oversold < 30, overbought > 70
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Chop regime: chop > 61.8 = range (mean revert), chop < 38.2 = trending
        chop_range = chop_aligned[i] > 61.8
        chop_trend = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long when: uptrend + oversold + ranging market (mean reversion setup)
            if kama_trend == 1 and rsi_oversold and chop_range:
                signals[i] = 0.25
                position = 1
            # Short when: downtrend + overbought + ranging market (mean reversion setup)
            elif kama_trend == -1 and rsi_overbought and chop_range:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when trend changes or overbought
            if kama_trend == -1 or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when trend changes or oversold
            if kama_trend == 1 or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals