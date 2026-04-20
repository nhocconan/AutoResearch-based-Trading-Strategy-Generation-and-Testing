#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF analysis
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d KAMA for trend direction (ER=10)
    close_s = pd.Series(close_1d)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = [close_1d[0]]
    for i in range(1, len(close_1d)):
        kama.append(kama[-1] + sc.iloc[i] * (close_1d[i] - kama[-1]))
    kama = np.array(kama)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # 1d RSI(14) for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d Chopiness Index for regime detection (14-period)
    atr1 = np.abs(high_1d - low_1d)
    atr2 = np.abs(high_1d - np.roll(close_1d, 1))
    atr2[0] = atr1[0]
    atr3 = np.abs(low_1d - np.roll(close_1d, 1))
    atr3[0] = atr1[0]
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)
    
    # Align HTF indicators to 12h timeframe
    kama_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_slope)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h price data
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(kama_slope_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range filter: avoid extreme chop values
        if chop_aligned[i] > 61.8:  # High chop = range market
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        kama_up = kama_slope_aligned[i] > 0
        rsi_not_extreme = 40 < rsi_aligned[i] < 60
        
        if position == 0:
            # Long when KAMA trending up and RSI in neutral zone
            if kama_up and rsi_not_extreme:
                signals[i] = 0.25
                position = 1
            # Short when KAMA trending down and RSI in neutral zone
            elif not kama_up and rsi_not_extreme:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down or RSI becomes overbought
            if (not kama_up) or (rsi_aligned[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up or RSI becomes oversold
            if kama_up or (rsi_aligned[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_KAMA_RSI_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0