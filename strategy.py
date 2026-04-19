#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_KAMA_RSI_ChopFilter_V2"
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
    
    # Get 1d data for Chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Chop index on 1d timeframe (14-period)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop_denom = highest_high - lowest_low
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_1d = 100 * np.log10(np.sum(tr_1d) / chop_denom) / np.log10(14)
    chop_1d = np.where(chop_denom == 1e-10, 50, chop_1d)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate KAMA on 12h timeframe
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))
    change = np.concatenate([[np.nan]*10, change])
    volatility = np.abs(np.diff(close))
    volatility = np.concatenate([[np.nan], volatility])
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI on 12h timeframe
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
            
        # Chop filter: range market (Chop > 61.8) for mean reversion
        chop_condition = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long when price crosses above KAMA and RSI oversold in ranging market
            if close[i] > kama[i] and rsi[i] < 30 and chop_condition:
                signals[i] = 0.25
                position = 1
            # Short when price crosses below KAMA and RSI overbought in ranging market
            elif close[i] < kama[i] and rsi[i] > 70 and chop_condition:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below KAMA or RSI overbought
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above KAMA or RSI oversold
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals